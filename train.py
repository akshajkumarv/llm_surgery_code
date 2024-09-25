import os
import time
import argparse
import warnings
import torch
import functools
import yaml
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data.distributed import DistributedSampler    

from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM 
from dataset import WikiBio, WebText10k, ARCEasy, tokenize, perplexity, accuracy, accuracy_arc_easy
from utils.train_utils import (
    get_local_rank,
    get_rank,
    get_world_size,
    setup,
    cleanup,
    get_device,
    write_log,
    report,
    param_count,
    get_model,
    get_cosine_schedule_with_warmup_lr_lambda,
    get_param_groups,
    get_fsdp_policies,
    CheckpointManager,
    #DataLoaderWrapper,
    compute_kl,
)

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

parser = argparse.ArgumentParser(description="Open Llama training")
parser.add_argument("--config", type=str, default=None, help="Config file")
args = parser.parse_args()
config_filename = args.config
assert config_filename is not None, "Config file cannot be empty."

assert os.path.isfile(
    config_filename
), "Cannot locate config file. Please check the path and try again."


with open(config_filename) as fp:
    config = yaml.safe_load(fp)

# parse the configurations for various sections
data_config = config["dataloader"]
model_config = config["model"]
optimizer_config = config["optimizer"]
train_config = config["train"]
sharding_config = config["sharding"]
scheduler_config = config["scheduler"]

# Reproducability
seed = int(train_config["seed"])
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# set by default to use Ampere tensor cores
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

train_config["ckp_dir"] = f"bs_{train_config['bsize']}_wd_{optimizer_config['weight_decay']}_lr_{optimizer_config['lr']}_warmup_{train_config['num_warmup_steps']}_steps_{train_config['num_training_steps']}_{train_config['unlearn_weight']}_{train_config['edit_unlearn_weight']}_{train_config['edit_update_weight']}_{train_config['retain_weight']}/"
train_config["log_dir"] = f"bs_{train_config['bsize']}_wd_{optimizer_config['weight_decay']}_lr_{optimizer_config['lr']}_warmup_{train_config['num_warmup_steps']}_steps_{train_config['num_training_steps']}_{train_config['unlearn_weight']}_{train_config['edit_unlearn_weight']}_{train_config['edit_update_weight']}_{train_config['retain_weight']}/"
log_path = os.path.join(train_config["root_dir"], f'{train_config["log_dir"]}logs/')
ckp_path = os.path.join(train_config["root_dir"], f'{train_config["ckp_dir"]}checkpoints/')
ckp_read_path = None
if "ckp_read_dir" in train_config.keys():
    ckp_read_path = train_config["ckp_read_dir"]

def train_model():
    start = time.time()
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    device = get_device(None, local_rank)
    torch.cuda.set_device(local_rank)

    print(
        f"rank={rank}, local_rank={local_rank}, world_size = {world_size}, device={device}"
    )
    #report("Starting distributed setup")
    setup()
    #report("NCCL setup complete")
    # print(f'rank={dist.get_global_rank()}, local_rank={dist.get_rank()}, world_size = {dist.get_world_size()}, device={device}')
    bsize = int(train_config["bsize"])
    effective_bsize = bsize * world_size
    report(effective_batch_size=effective_bsize)
    start_step = 0

    # print out the configuration
    if rank == 0:
        os.makedirs(log_path, exist_ok=True)
        write_log(log_path, tb_writer=None, **config)
    
    # Load the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(f"{train_config['model_path']}")
    tokenizer.pad_token = tokenizer.eos_token

    unlearn = WikiBio(data_config, split="unlearn")
    unlearn_sampler = DistributedSampler(unlearn, rank=rank, num_replicas=world_size, shuffle=data_config["shuffle"], drop_last=data_config["drop_last"])
    unlearn_loader = torch.utils.data.DataLoader(unlearn, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                 pin_memory=data_config["pin_memory"], sampler=unlearn_sampler)
    unlearn_loader_iter = iter(unlearn_loader)
    
    edit_unlearn = WikiBio(data_config, split="edit_unlearn")
    edit_unlearn_sampler = DistributedSampler(edit_unlearn, rank=rank, num_replicas=world_size, shuffle=data_config["shuffle"], drop_last=data_config["drop_last"])
    edit_unlearn_loader = torch.utils.data.DataLoader(edit_unlearn, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                      pin_memory=data_config["pin_memory"], sampler=edit_unlearn_sampler)
    edit_unlearn_loader_iter = iter(edit_unlearn_loader)
    
    edit_update = WikiBio(data_config, split="edit_update")
    edit_update_sampler = DistributedSampler(edit_update, rank=rank, num_replicas=world_size, shuffle=data_config["shuffle"], drop_last=data_config["drop_last"])
    edit_update_loader = torch.utils.data.DataLoader(edit_update, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                     pin_memory=data_config["pin_memory"], sampler=edit_update_sampler)
    edit_update_loader_iter = iter(edit_update_loader)
    
    retain = WebText10k(data_config, train_split=True, eval_split=False)
    retain_sampler = DistributedSampler(retain, rank=rank, num_replicas=world_size, shuffle=data_config["shuffle"], drop_last=data_config["drop_last"])
    retain_loader = torch.utils.data.DataLoader(retain, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                pin_memory=data_config["pin_memory"], sampler=retain_sampler)
    retain_loader_iter = iter(retain_loader)

    val_unlearn = WikiBio(data_config, split="val_unlearn")
    val_unlearn_sampler = DistributedSampler(val_unlearn, rank=rank, num_replicas=world_size, shuffle=False, drop_last=data_config["drop_last"])
    val_unlearn_loader = torch.utils.data.DataLoader(val_unlearn, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                     pin_memory=data_config["pin_memory"], sampler=val_unlearn_sampler)
    
    val_edit_unlearn = WikiBio(data_config, split="val_edit_unlearn")
    val_edit_unlearn_sampler = DistributedSampler(val_edit_unlearn, rank=rank, num_replicas=world_size, shuffle=False, drop_last=data_config["drop_last"])
    val_edit_unlearn_loader = torch.utils.data.DataLoader(val_edit_unlearn, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                          pin_memory=data_config["pin_memory"], sampler=val_edit_unlearn_sampler)
    
    val_edit_update = WikiBio(data_config, split="val_edit_update")
    val_edit_update_sampler = DistributedSampler(val_edit_update, rank=rank, num_replicas=world_size, shuffle=False, drop_last=data_config["drop_last"])
    val_edit_update_loader = torch.utils.data.DataLoader(val_edit_update, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                         pin_memory=data_config["pin_memory"], sampler=val_edit_update_sampler)
        
    val_retain = WebText10k(data_config, train_split=False, eval_split='validation')
    val_retain_sampler = DistributedSampler(val_retain, rank=rank, num_replicas=world_size, shuffle=False, drop_last=data_config["drop_last"])
    val_retain_loader = torch.utils.data.DataLoader(val_retain, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                    pin_memory=data_config["pin_memory"], sampler=val_retain_sampler)
    
    val_benchmark = ARCEasy(data_config, split="validation")
    val_benchmark_sampler = DistributedSampler(val_benchmark, rank=rank, num_replicas=world_size, shuffle=False, drop_last=data_config["drop_last"])
    val_benchmark_loader = torch.utils.data.DataLoader(val_benchmark, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                       pin_memory=data_config["pin_memory"], sampler=val_benchmark_sampler)
    
    test_unlearn = WikiBio(data_config, split="test_unlearn")
    test_unlearn_sampler = DistributedSampler(test_unlearn, rank=rank, num_replicas=world_size, shuffle=False, drop_last=data_config["drop_last"])
    test_unlearn_loader = torch.utils.data.DataLoader(test_unlearn, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                      pin_memory=data_config["pin_memory"], sampler=test_unlearn_sampler)
    
    test_edit_unlearn = WikiBio(data_config, split="test_edit_unlearn")
    test_edit_unlearn_sampler = DistributedSampler(test_edit_unlearn, rank=rank, num_replicas=world_size, shuffle=False, drop_last=data_config["drop_last"])
    test_edit_unlearn_loader = torch.utils.data.DataLoader(test_edit_unlearn, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                           pin_memory=data_config["pin_memory"], sampler=test_edit_unlearn_sampler)
    
    test_edit_update = WikiBio(data_config, split="test_edit_update")
    test_edit_update_sampler = DistributedSampler(test_edit_update, rank=rank, num_replicas=world_size, shuffle=False, drop_last=data_config["drop_last"])
    test_edit_update_loader = torch.utils.data.DataLoader(test_edit_update, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                          pin_memory=data_config["pin_memory"], sampler=test_edit_update_sampler)
        
    test_retain = WebText10k(data_config, train_split=False, eval_split='test')
    test_retain_sampler = DistributedSampler(test_retain, rank=rank, num_replicas=world_size, shuffle=False, drop_last=data_config["drop_last"])
    test_retain_loader = torch.utils.data.DataLoader(test_retain, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                     pin_memory=data_config["pin_memory"], sampler=test_retain_sampler)

    test_benchmark = ARCEasy(data_config, split="test")
    test_benchmark_sampler = DistributedSampler(test_benchmark, rank=rank, num_replicas=world_size, shuffle=False, drop_last=data_config["drop_last"])
    test_benchmark_loader = torch.utils.data.DataLoader(test_benchmark, batch_size=train_config["bsize"], num_workers=data_config["num_workers"], 
                                                       pin_memory=data_config["pin_memory"], sampler=test_benchmark_sampler)
    
    checkpointer = CheckpointManager(ckp_path, sharding_config["sharding_strategy"])
    
    # create model
    #report("Constructing model...")
    model = LlamaForCausalLM.from_pretrained(f"{train_config['model_path']}", **model_config)
    #report("Model constructed")
    #report(f"nparams = {param_count(model)}")

    if train_config["enable_grad_checkpoint"]:
        model.gradient_checkpointing_enable()
        #report("Gradient Checkpointing enabled")
    
    #report("Applying FSDP wrapper..")
    fsdp_policies = get_fsdp_policies(sharding_config)
    model = FSDP(
        model,
        auto_wrap_policy=fsdp_policies["wrapping_policy"],
        mixed_precision=fsdp_policies["mp_policy"],
        sharding_strategy=fsdp_policies["sharding_policy"],
        device_id=local_rank,
        limit_all_gathers=sharding_config["limit_all_gathers"],
        use_orig_params=sharding_config["use_orig_params"],
    )
    model.to(local_rank)

    pretrained_model = LlamaForCausalLM.from_pretrained("/fsx/sambits_interns_2024/model_zoo/meta-llama_Llama-2-7b-hf", **model_config)
    pretrained_model = FSDP(
        pretrained_model,
        auto_wrap_policy=fsdp_policies["wrapping_policy"],
        mixed_precision=fsdp_policies["mp_policy"],
        sharding_strategy=fsdp_policies["sharding_policy"],
        device_id=local_rank,
        limit_all_gathers=sharding_config["limit_all_gathers"],
        use_orig_params=True,
    )  
    pretrained_model.to(local_rank)
    
    if rank == 0:
        print("###### FSDP wrapped model #######")
        print(model)

    #report("Initializing optimizer")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=optimizer_config["weight_decay"],
        lr=float(optimizer_config["lr"]),
        betas=(float(optimizer_config["beta1"]), float(optimizer_config["beta2"])),
        eps=float(optimizer_config["eps"]),
    )
    #report("Optimizer initialization complete")
    seq_len = int(train_config["seq_len"])
    
    torch.cuda.empty_cache()
    
    if ckp_read_path is not None:
        report("Loading checkpoints...")
        no_optim = True
        no_dataloader = False
        if "no_optim" in optimizer_config.keys():
            no_optim = optimizer_config["no_optim"]
        if "no_dataloader" in optimizer_config.keys():
            no_dataloader = optimizer_config["no_dataloader"]

        single_file_ckp_name = train_config.get("single_file_ckp", None)
        single_ckp_path = None
        if single_file_ckp_name is not None:
            single_ckp_path = os.path.join(ckp_read_path, single_file_ckp_name)
            assert os.path.isfile(single_ckp_path), "Specified checkpoint is not a file."

            report("Loading single file checkpoint...")

        dist.barrier()
        metadata = checkpointer.load_ckp(
            ckp_read_path,
            model,
            no_optim=no_optim,
        )
        report("Finished loading model checkpoints.")
        dist.barrier()
        model.train()

        dist.barrier()
        pretrained__model_metadata = checkpointer.load_ckp(
            ckp_read_path,
            pretrained_model,
            no_optim=no_optim,
        )
        report("Finished loading pretrained-model checkpoints.")
        dist.barrier()
        for n, p in pretrained_model.named_parameters():
            p.requires_grad = False
        dist.barrier()
        pretrained_model.eval()

    #report("Initializing scheduler")
    ### Keeping it Unchanged
    lr_lambda = functools.partial(
        get_cosine_schedule_with_warmup_lr_lambda,
        start_step=start_step,
        num_warmup_steps=int(train_config["num_warmup_steps"]),
        num_training_steps=int(train_config["num_training_steps"]),
        num_cycles=float(scheduler_config["num_cycles"]),
        max_lr=float(scheduler_config["max_lr"]),
        min_lr=float(scheduler_config["min_lr"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    #report("Scheduler initialized")

    # tracked variables
    loss_acc = torch.zeros(1).to(local_rank)
    ### train_start_time = time.time()

    #report("Entering training loop")
    torch.cuda.synchronize()
    start_interval = time.time()
    sub_step_count = 0

    if rank == 0:
        os.makedirs(log_path, exist_ok=True)
        writer = None
        if train_config["log_to_tb"]:
            tb_log_path = os.path.join(log_path, "tb_logs/")
            os.makedirs(tb_log_path, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_log_path)

    # data_iter = iter(data_loader)
    effective_training_steps = int(train_config["num_training_steps"]) - start_step
    assert (
        effective_training_steps > 0
    ), "Increase num_training_steps as checkpoint already trained for longer number of steps"
    #report(effective_training_steps=effective_training_steps)
    unlearn_count = 0
    edit_unlearn_count = 0
    edit_update_count = 0
    retain_count = 0
    report(train_config['log_dir'])
    for step in range(start_step, int(train_config["num_training_steps"])):
        optimizer.zero_grad()
        if train_config["unlearn_weight"] > 0.0:
            # if rank == 0:
            #     print("Yes Unlearn")
            try:
                unlearn_batch = next(unlearn_loader_iter)
            except StopIteration:
                unlearn_count += 1
                unlearn_sampler.set_epoch(unlearn_count)
                unlearn_loader_iter = iter(unlearn_loader)
                unlearn_batch = next(unlearn_loader_iter)
            unlearn_batch = tokenize(unlearn_batch, tokenizer, local_rank, test=False)
            unlearn_output = model(**unlearn_batch)
            unlearn_loss = unlearn_output.loss
        else:
            # if rank == 0:
            #     print("No Unlearn")
            unlearn_loss = 0.0
        
        if train_config["edit_unlearn_weight"] > 0.0:
            # if rank == 0:
            #     print("Yes Edit Unlearn")
            try:
                edit_unlearn_batch = next(edit_unlearn_loader_iter)
            except StopIteration:
                edit_unlearn_count += 1
                edit_unlearn_sampler.set_epoch(edit_unlearn_count)
                edit_unlearn_loader_iter = iter(edit_unlearn_loader)
                edit_unlearn_batch = next(edit_unlearn_loader_iter)
            edit_unlearn_batch = tokenize(edit_unlearn_batch, tokenizer, local_rank, test=False)
            edit_unlearn_output = model(**edit_unlearn_batch)
            edit_unlearn_loss = edit_unlearn_output.loss
        else:
            # if rank == 0:
            #     print("No Edit Unlearn")
            edit_unlearn_loss = 0.0

        if train_config["edit_update_weight"] > 0.0:
            # if rank == 0:
            #     print("Yes Edit Update")
            try:
                edit_update_batch = next(edit_update_loader_iter)
            except StopIteration:
                edit_update_count += 1
                edit_update_sampler.set_epoch(edit_update_count)
                edit_update_loader_iter = iter(edit_update_loader)
                edit_update_batch = next(edit_update_loader_iter)
            edit_update_batch = tokenize(edit_update_batch, tokenizer, local_rank, test=False)
            edit_update_output = model(**edit_update_batch)
            edit_update_loss = edit_update_output.loss
        else:
            # if rank == 0:
            #     print("No Edit Update")
            edit_update_loss = 0.0

        if train_config["retain_weight"] > 0.0:
            # if rank == 0:
            #     print("Yes Retain")
            try:
                retain_batch = next(retain_loader_iter)
            except StopIteration:
                retain_count += 1
                retain_sampler.set_epoch(retain_count)
                retain_loader_iter = iter(retain_loader)
                retain_batch = next(retain_loader_iter)
            retain_batch = tokenize(retain_batch, tokenizer, local_rank, test=False)
            retain_loss = compute_kl(retain_batch, model, pretrained_model)
        else:
            # if rank == 0:
            #     print("No Retain")
            retain_loss = 0.0

        loss = (
            - (train_config["unlearn_weight"] * unlearn_loss) 
            - (train_config["edit_unlearn_weight"] * edit_unlearn_loss) 
            + (train_config["edit_update_weight"] * edit_update_loss)
            + (train_config["retain_weight"] * retain_loss)
        )

        #if step == start_step:
            #report("Completed first forward pass")

        loss_acc += loss.item()
        loss.backward()
        
        #if step == start_step:
            #report("Completed first backward pass")

        grad_norm = model.clip_grad_norm_(float(train_config["clip_th"])).item()
        
        optimizer.step()
        scheduler.step()
        sub_step_count += 1
        
        if ((step + 1) % int(train_config["report_interval"])) == 0:
            dist.all_reduce(loss_acc, op=dist.ReduceOp.SUM)
            trainloss = loss_acc.item() / world_size / sub_step_count
        
            torch.cuda.synchronize()
            step_time = (time.time() - start_interval) / sub_step_count
            
            dist.barrier()
            model.eval()

            dist.barrier()
            total_ppl, num_examples = perplexity(model, tokenizer, unlearn_loader, local_rank, test=False)
            dist.all_reduce(total_ppl, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
            unlearn_ppl = (total_ppl/num_examples).item()
    
            dist.barrier()
            total_ppl, num_examples = perplexity(model, tokenizer, edit_unlearn_loader, local_rank, test=False)
            dist.all_reduce(total_ppl, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
            edit_unlearn_ppl = (total_ppl/num_examples).item()
   
            dist.barrier()
            total_ppl, num_examples = perplexity(model, tokenizer, edit_update_loader, local_rank, test=False)
            dist.all_reduce(total_ppl, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
            edit_update_ppl = (total_ppl/num_examples).item()
    
            dist.barrier()
            total_ppl, num_examples = perplexity(model, tokenizer, retain_loader, local_rank, test=False)
            dist.all_reduce(total_ppl, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
            retain_ppl = (total_ppl/num_examples).item()
    
            dist.barrier()
            model.train()
            
            step_log = {
                "step": step + 1,
                "trainloss": trainloss,
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm,
                "avg_step_time_secs": step_time,
                "Unlearn PPL": round(unlearn_ppl, 2),
                "Edit Unlearn PPL": round(edit_unlearn_ppl, 2),
                "Edit Update PPL": round(edit_update_ppl, 2),
                "Retain PPL": round(retain_ppl, 2),
            }
            #report(**step_log)
            if rank == 0:
                write_log(log_path, tb_writer=writer, **step_log)

            # reset variables
            start_interval = time.time()
            sub_step_count = 0
            loss_acc.zero_()

    report("Outside training loop.")
    
    dist.barrier()
    model.eval()

    dist.barrier()
    total_ppl, num_examples = perplexity(model, tokenizer, unlearn_loader, local_rank, test=False)
    dist.all_reduce(total_ppl, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    unlearn_ppl = (total_ppl/num_examples).item()
    if rank == 0:
        print("="*100)
        report(train_config['log_dir'])
        print(f"Unlearn PPL: {round(unlearn_ppl, 2)}")
    
    dist.barrier()
    total_ppl, num_examples = perplexity(model, tokenizer, edit_unlearn_loader, local_rank, test=False)
    dist.all_reduce(total_ppl, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    edit_unlearn_ppl = (total_ppl/num_examples).item()
    if rank == 0:
        print(f"Edit Unlearn PPL: {round(edit_unlearn_ppl, 2)}")

    dist.barrier()
    total_ppl, num_examples = perplexity(model, tokenizer, edit_update_loader, local_rank, test=False)
    dist.all_reduce(total_ppl, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    edit_update_ppl = (total_ppl/num_examples).item()
    if rank == 0:
        print(f"Edit Update PPL: {round(edit_update_ppl, 2)}")

    dist.barrier()
    total_ppl, num_examples = perplexity(model, tokenizer, retain_loader, local_rank, test=False)
    dist.all_reduce(total_ppl, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    retain_ppl = (total_ppl/num_examples).item()
    if rank == 0:
        print(f"Retain PPL: {round(retain_ppl, 2)}")

    dist.barrier()
    correct, num_examples = accuracy(model, tokenizer, val_unlearn_loader, local_rank, test=True)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    val_unlearn_acc = (correct*100/num_examples).item()
    if rank == 0:
        print(f"Validation Unlearn Accuracy: {round(val_unlearn_acc, 2)}")

    dist.barrier()
    correct, num_examples = accuracy(model, tokenizer, val_edit_unlearn_loader, local_rank, test=True)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    val_edit_unlearn_acc = (correct*100/num_examples).item()
    if rank == 0:
        print(f"Validation Edit Unlearn Accuracy: {round(val_edit_unlearn_acc, 2)}")

    dist.barrier()
    correct, num_examples = accuracy(model, tokenizer, val_edit_update_loader, local_rank, test=True)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    val_edit_update_acc = (correct*100/num_examples).item()
    if rank == 0:
        print(f"Validation Edit Update Accuracy: {round(val_edit_update_acc, 2)}")
    
    dist.barrier()
    correct, num_examples = accuracy(model, tokenizer, val_retain_loader, local_rank, test=True)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    val_retain_acc = (correct*100/num_examples).item()
    if rank == 0:
        print(f"Validation Retain Accuracy: {round(val_retain_acc, 2)}")
    
    dist.barrier()
    correct, num_examples = accuracy_arc_easy(model, tokenizer, val_benchmark_loader, local_rank, test=True)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    val_benchmark_acc = (correct*100/num_examples).item()
    if rank == 0:
        print(f"Validation Benchmark Accuracy: {round(val_benchmark_acc, 2)}")
    
    dist.barrier()
    correct, num_examples = accuracy(model, tokenizer, test_unlearn_loader, local_rank, test=True)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    test_unlearn_acc = (correct*100/num_examples).item()
    if rank == 0:
        print(f"Test Unlearn Accuracy: {round(test_unlearn_acc, 2)}")
    
    dist.barrier()
    correct, num_examples = accuracy(model, tokenizer, test_edit_unlearn_loader, local_rank, test=True)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    test_edit_unlearn_acc = (correct*100/num_examples).item()
    if rank == 0:
        print(f"Test Edit Unlearn Accuracy: {round(test_edit_unlearn_acc, 2)}")

    dist.barrier()
    correct, num_examples = accuracy(model, tokenizer, test_edit_update_loader, local_rank, test=True)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    test_edit_update_acc = (correct*100/num_examples).item()
    if rank == 0:
        print(f"Test Edit Update Accuracy: {round(test_edit_update_acc, 2)}")
    
    dist.barrier()
    correct, num_examples = accuracy(model, tokenizer, test_retain_loader, local_rank, test=True)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    test_retain_acc = (correct*100/num_examples).item()
    if rank == 0:
        print(f"Test Retain Accuracy: {round(test_retain_acc, 2)}")
    
    dist.barrier()
    correct, num_examples = accuracy_arc_easy(model, tokenizer, test_benchmark_loader, local_rank, test=True)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_examples, op=dist.ReduceOp.SUM)
    test_benchmark_acc = (correct*100/num_examples).item()
    if rank == 0:
        print(f"Test Benchmark Accuracy: {round(test_benchmark_acc, 2)}")
        
    dist.barrier()
    if rank == 0:
        final_results = {
            "Unlearn PPL": round(unlearn_ppl, 2),
            "Edit Unlearn PPL": round(edit_unlearn_ppl, 2),
            "Edit Update PPL": round(edit_update_ppl, 2),
            "Retain PPL": round(retain_ppl, 2),
            "Validation Unlearn Accuracy": round(val_unlearn_acc, 2),
            "Validation Edit Unlearn Accuracy": round(val_edit_unlearn_acc, 2),
            "Validation Edit Update Accuracy": round(val_edit_update_acc, 2),
            "Validation Retain Accuracy": round(val_retain_acc, 2),
            "Validation Benchmark Accuracy": round(val_benchmark_acc, 2),
            "Test Unlearn Accuracy": round(test_unlearn_acc, 2),
            "Test Edit Unlearn Accuracy": round(test_edit_unlearn_acc, 2),
            "Test Edit Update Accuracy": round(test_edit_update_acc, 2),
            "Test Retain Accuracy": round(test_retain_acc, 2),
            "Test Benchmark Accuracy": round(test_benchmark_acc, 2)
        }
        write_log(log_path, tb_writer=None, **final_results)
    
    dist.barrier()
    report("Writing final checkpoint..")
    checkpointer.save_ckp(
        step + 1, model, ### optimizer, ###data_loader, 
        loss=trainloss, ###tokens_seen=tokens_seen
    )
    report("Finished writing final checkpoint")
    report(train_config['log_dir'])
    cleanup()
    report("Cleanup complete")
    if rank == 0:
        print(f"Total training time (in hours): {(time.time() - start)/3600.0}")
        print("="*100)

if __name__ == "__main__":
    train_model()

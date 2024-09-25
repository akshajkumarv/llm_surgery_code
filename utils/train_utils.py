import os
import math
import json
import functools
import torch
import torch.distributed as dist
from transformers import LlamaConfig, LlamaForCausalLM
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed._shard.checkpoint import (
    FileSystemWriter,
    FileSystemReader,
    SavePlan,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaDecoderLayer
from streaming import StreamingDataset
from omegaconf import OmegaConf as om
from text_data_stream import build_text_dataloader


def get_local_rank():
    return int(os.getenv("LOCAL_RANK"))


def get_rank():
    return int(os.getenv("RANK"))


def get_world_size():
    return int(os.getenv("WORLD_SIZE"))


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def get_device(device_type, local_rank):
    if device_type is None:
        if torch.cuda.is_available():
            device_type = "cuda"
        elif torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    return torch.device(device_type, local_rank)


def write_log(log_path, tb_writer=None, key="run", **kwargs):
    if len(kwargs) > 0:
        log_file = os.path.join(log_path, "log.json")
        if not os.path.exists(log_file):
            mode = "w"
        else:
            mode = "a"
        with open(log_file, mode) as fp:
            json.dump(kwargs, fp)
            fp.write("\n")
        fp.close()

        if tb_writer is not None:
            # key = 'Step_'+str(kwargs['step'])
            # TODO: change 'key' to "step" later on to group plots based on "step" key
            tb_writer.add_scalars(key, kwargs, kwargs["step"])
            ###tb_writer.add_scalars("token", kwargs, kwargs["tokens_seen_in_training"])
            tb_writer.flush()


def report(*args, **kwargs):
    rank = get_rank()
    if rank == 0:
        print(*args)
        for key in kwargs:
            print(f"{key} = {kwargs[key]}")


def param_count(model):
    return f"{sum([torch.numel(p) for p in model.parameters()]):,}"


def get_model(model_config):
    m_config = LlamaConfig(**model_config)
    model = LlamaForCausalLM(m_config)
    return model


def get_cosine_schedule_with_warmup_lr_lambda(
    current_step,
    start_step,
    num_warmup_steps,
    num_training_steps,
    num_cycles,
    max_lr,
    min_lr,
):
    current_step += start_step
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        0.0,
        min_lr
        + 0.5
        * float(max_lr - min_lr)
        * (1.0 + math.cos(float(num_cycles) * 2.0 * progress * math.pi)),
    )


def get_param_groups(model, optimizer_config):
    PARAM_GROUP_FIELDS = ("param_names",)
    decay = set()
    no_decay = set()
    all_params = {}

    decay_norm_and_bias = optimizer_config["decay_norm_and_bias"]
    decay_embeddings = optimizer_config["decay_embeddings"]

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters():
            if not param.requires_grad:
                continue

            full_param_name = (
                f"{module_name}.{param_name}" if module_name else param_name
            )
            all_params[full_param_name] = param

            if param_name.endswith("bias"):
                if decay_norm_and_bias:
                    decay.add(full_param_name)
                else:
                    no_decay.add(full_param_name)
            elif param_name.endswith("weight") and isinstance(module, torch.nn.Linear):
                decay.add(full_param_name)
            elif param_name.endswith("weight") and isinstance(
                module, (LlamaRMSNorm, torch.nn.LayerNorm)
            ):
                if decay_norm_and_bias:
                    decay.add(full_param_name)
                else:
                    no_decay.add(full_param_name)
            elif param_name.endswith("weight") and isinstance(
                module, torch.nn.Embedding
            ):
                if decay_embeddings:
                    decay.add(full_param_name)
                else:
                    no_decay.add(full_param_name)
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(all_params.keys() - union_params) == 0
    ), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

    # Create the pytorch optimizer groups.
    decay_sorted = sorted(list(decay))
    no_decay_sorted = sorted(list(no_decay))
    param_groups = []

    if len(decay_sorted) > 0:
        param_groups.append(
            {
                "params": [all_params[param_name] for param_name in decay_sorted],
                "param_names": decay_sorted,
            }
        )
    if len(no_decay_sorted) > 0:
        param_groups.append(
            {
                "params": [all_params[param_name] for param_name in no_decay_sorted],
                "param_names": no_decay_sorted,
                "weight_decay": 0.0,
            }
        )

    # Validate fields.
    for group in param_groups:
        for key in PARAM_GROUP_FIELDS:
            assert key in group

    return param_groups


def get_fsdp_policies(sharding_config):
    """
    Pass FSDP configs like sharding policy, mixedPrecision policy, and wrapping policy
    """
    dtype_mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "quint8": torch.quint8,
        "qint8": torch.qint8,
        "qint32": torch.qint32,
        "quint4x2": torch.quint4x2,
    }
    param_dtype_str, reduce_dtype_str, buffer_dtype_str = (
        sharding_config["param_dtype"],
        sharding_config["reduce_dtype"],
        sharding_config["buffer_dtype"],
    )
    model_sharding_strategies = {
        "fsdp": ShardingStrategy.FULL_SHARD,
        "hsdp": ShardingStrategy.HYBRID_SHARD,
        "grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    sharding_policy = model_sharding_strategies.get(
        sharding_config["sharding_strategy"]
    )
    mp_policy = (
        None
        if not sharding_config["use_bf16"]
        else MixedPrecision(
            param_dtype=dtype_mapping.get(param_dtype_str),
            reduce_dtype=dtype_mapping.get(reduce_dtype_str),
            buffer_dtype=dtype_mapping.get(buffer_dtype_str),
        )
    )
    wrapping_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}
    )
    return {
        "wrapping_policy": wrapping_policy,
        "mp_policy": mp_policy,
        "sharding_policy": sharding_policy,
    }


class CheckpointManager:
    def __init__(self, ckp_path=None, sharding_strategy="fsdp"):
        super(CheckpointManager, self).__init__()
        self.sharding_strategy = sharding_strategy
        self.ckp_path = ckp_path
        if self.ckp_path is not None:
            os.makedirs(self.ckp_path, exist_ok=True)

    def save_ckp(self, step, model, optimizer=None, dataloader=None, **kwargs):
        assert self.ckp_path is not None, "Please specify a ckp save directory"
        rank = get_rank()
        local_rank = get_local_rank()

        model_state = None
        optim_state = None
        dataloader_state = None

        if model is not None:
            if (self.sharding_strategy == "no_shard") and (rank == 0):
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    model_state = model.state_dict()
                    if optimizer is not None:
                        optim_state = FSDP.optim_state_dict(model, optimizer)
            else:
                with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                    model_state = model.state_dict()
                    if optimizer is not None:
                        optim_state = FSDP.optim_state_dict(
                            model,
                            optimizer,
                            # model, optimizer, group=model.process_group
                        )
        if dataloader is not None:
            dataloader_state = dataloader.get_state_dict()

        step_ckp_dir = os.path.join(self.ckp_path, "step_" + str(step))
        os.makedirs(step_ckp_dir, exist_ok=True)
        writer = FileSystemWriter(step_ckp_dir, single_file_per_rank=True)

        ckp_state_dict = None
        if (
            (self.sharding_strategy == "fsdp")
            or ((self.sharding_strategy == "hsdp") and (rank == local_rank))
            or ((self.sharding_strategy == "no_shard") and (rank == 0))
        ):
            ckp_state_dict = {"model": model_state, "optimizer": optim_state}
        if ckp_state_dict is not None and model is not None:
            if (self.sharding_strategy == "no_shard") and (rank == 0):
                torch.save(ckp_state_dict, os.path.join(step_ckp_dir, "model_opt.pth"))
            else:
                save_state_dict(
                    state_dict=ckp_state_dict,
                    storage_writer=writer,
                    process_group=model.process_group,
                    planner=DefaultSavePlanner(),
                )

        # save the data loader separately from the pytorch specific state_dicts
        if dataloader_state is not None:
            torch.save(
                dataloader_state, os.path.join(step_ckp_dir, f"dataloader_{rank}.pth")
            )

        if rank == 0:
            metadata = kwargs
            metadata["step"] = step
            torch.save(metadata, os.path.join(step_ckp_dir, "meta.pth"))

    def load_ckp(self, ckp_read_path, model, optimizer=None, dataloader=None, **kwargs):
        rank = get_rank()
        if not os.path.exists(ckp_read_path):
            if rank == 0:
                report(
                    "Skipping because no sharded_state_dict checkpoint directory found"
                )
            return
        reader = FileSystemReader(ckp_read_path)

        single_ckp_path = kwargs.get("single_ckp_path", None)

        if (single_ckp_path is not None) or (self.sharding_strategy == "no_shard"):
            # in case of no_shard, if no file name is given assume default name
            if single_ckp_path is None:
                single_ckp_path = os.path.join(ckp_read_path, "model_opt.pth")

            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                report(f"Loading model checkpoints from {single_ckp_path}")
                checkpoint_noshard = torch.load(single_ckp_path)

                model.load_state_dict(checkpoint_noshard["model"])

                report(f"Model successfully loaded from {single_ckp_path}\n")

                if ("no_optim" not in kwargs.keys()) or (not kwargs["no_optim"]):
                    report(f"Loading optimizer from {single_ckp_path}")
                    flattened_osd = FSDP.optim_state_dict_to_load(
                        model=model,
                        optim=optimizer,
                        optim_state_dict=checkpoint_noshard["optimizer"],
                    )
                    optimizer.load_state_dict(flattened_osd)
                    report(f"Optimizer loaded from {single_ckp_path}")
                else:
                    report(f"Optimizer state not loaded")
        else:
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                report(f"Loading model checkpoints from {ckp_read_path}")
                state_dict = {
                    "model": model.state_dict(),
                }
                load_state_dict(
                    state_dict=state_dict,
                    storage_reader=reader,
                    planner=DefaultLoadPlanner(),
                )
                model.load_state_dict(state_dict["model"])
                report(f"Model loaded from {ckp_read_path}\n")

                if ("no_optim" not in kwargs.keys()) or (not kwargs["no_optim"]):
                    report(f"Loading optimizer from {ckp_read_path}")
                    optim_state = load_sharded_optimizer_state_dict(
                        model_state_dict=model.state_dict(),
                        optimizer_key="optimizer",
                        storage_reader=reader,
                    )
                    flattened_osd = FSDP.optim_state_dict_to_load(
                        model=model,
                        optim=optimizer,
                        optim_state_dict=optim_state["optimizer"],
                    )
                    optimizer.load_state_dict(flattened_osd)
                    report(f"Optimizer loaded from {ckp_read_path}")
                else:
                    report(f"Optimizer state not loaded")

        dataloader_state = None

        if os.path.exists(os.path.join(ckp_read_path, f"dataloader_{rank}.pth")):
            if ("no_dataloader" not in kwargs.keys()) or (not kwargs["no_dataloader"]):
                dataloader_state = torch.load(
                    os.path.join(ckp_read_path, f"dataloader_{rank}.pth")
                )
                dataloader.set_state_dict(dataloader_state)
                report(f"Dataloader state loaded from {ckp_read_path}")
            else:
                report(f"Dataloader state not loaded")

        #report('loading metadata')
        #if os.path.exists(os.path.join(ckp_read_path, "meta.pth")):
        #    metadata = torch.load(os.path.join(ckp_read_path, "meta.pth"))
        #    return metadata

class DataLoaderWrapper(StreamingDataset):
    def __init__(self, bsize, tokenizer, data_config):
        self.data_loader = self.get_data_loader(bsize, tokenizer, data_config)
        self.data_iter = iter(self.data_loader)

    def get_next(self):
        try:
            data_seq = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            data_seq = next(self.data_iter)

        # We do not shift as HF forard function does it for us
        target = data_seq.clone()
        return data_seq, target

    def get_data_loader(self, bsize, tokenizer, data_config):
        cfg = om.create(data_config)
        return build_text_dataloader(
            cfg=cfg, tokenizer=tokenizer, device_batch_size=bsize
        )

    def get_state_dict(self):
        return self.data_loader.state_dict()

    def set_state_dict(self, state_dict):
        self.data_loader.load_state_dict(state_dict)


def compute_kl(batch, model, pretrained_model):
    outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])

    with torch.no_grad():
        pretrained_outputs = pretrained_model(batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(outputs.logits, -1)

    kl_loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return kl_loss

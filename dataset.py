import os
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
#nltk.download('punkt')
import ast

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset

class WebText10k(Dataset):
    # Reference: https://github.com/Thartvigsen/GRACE/blob/main/grace/dataset.py

    def __init__(self, config, train_split, eval_split):
        #self.indices_to_drop = [11, 72, 73, 100, 103, 107, 134, 153, 167, 170, 183, 235, 288, 307, 315, 320, 351, 372, 375, 379, 387, 397, 418, 449, 454, 477, 527, 533, 547, 627, 656, 719, 760, 791, 797, 802, 824, 878, 910, 942, 946, 1008]
        self.indices_to_drop = [11, 72, 73, 100, 103, 107, 134, 153, 167, 170, 183, 235, 288, 307, 315, 320, 351, 372, 375, 379, 387, 397, 418, 449, 454, 477, 527, 533, 547, 627, 656, 719, 760, 791, 797, 802, 824, 878, 910, 942, 946, 1008, 1046, 1093, 1105, 1121, 1125, 1239, 1256, 1281, 1330, 1339, 1380, 1389, 1393, 1399, 1450, 1522, 1617, 1644, 1652, 1670, 1677, 1699, 1714, 1734, 1754, 1797, 1813, 1845, 1879, 1917, 1942, 1953, 1960, 2029, 2036]
        if train_split:
            frac = int(config["retain_train_split"])
            data = pd.read_csv("./data/openwebtext-10k.csv")
            upstream = data["text"][:frac]
            upstream = self.drop_indices(upstream)
            self.text = [{"text": s, "labels": [], "concept": []} for s in upstream]
        else:
            frac = int(config["retain_test_split"])
            eval_data_owt10k = pd.read_csv("./data/gpt4_evaluation_benchmark_2200owt10k_mcqa.csv")
            eval_data_owt10k = eval_data_owt10k[:frac].copy().reset_index()
            eval_data_owt10k = self.rows_to_drop(eval_data_owt10k)
            self.text = []
            for _, row in eval_data_owt10k.iterrows():
                qa_pairs = ast.literal_eval(row['qa'])
                qa_pairs_val, qa_pairs_test = train_test_split(qa_pairs, shuffle=True, test_size=0.5, random_state=42)
                if eval_split == 'validation':
                    for qa in qa_pairs_val:
                        if not "option" in qa['correct answer']:
                            qa = self.replace_answer(qa)
                        self.text.append({"question":qa['Q'], "option1": qa['option1'], "option2": qa['option2'], "option3": qa['option3'], "labels": qa['correct answer']})
                elif eval_split == 'test':
                    for qa in qa_pairs_test:
                        if not "option" in qa['correct answer']:
                            qa = self.replace_answer(qa)
                        self.text.append({"question":qa['Q'], "option1": qa['option1'], "option2": qa['option2'], "option3": qa['option3'], "labels": qa['correct answer']})

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]
    
    def drop_indices(self, original_list):
        filtered_list = [item for idx, item in enumerate(original_list) if idx not in self.indices_to_drop]
        return filtered_list
    
    def rows_to_drop(self, df):
        df = df.drop(self.indices_to_drop)
        df = df.reset_index(drop=True)
        return df

    def replace_answer(self, qa):
        if not "option" in qa['correct answer']:
            if qa['correct answer'] == qa['option1']:
                qa['correct answer'] = 'option1'
            elif qa['correct answer'] == qa['option2']:
                qa['correct answer'] = 'option2'
            elif qa['correct answer'] == qa['option3']:
                qa['correct answer'] = 'option3'
        return qa

class WikiBio(Dataset):
    # Reference: https://github.com/Thartvigsen/GRACE/blob/main/grace/dataset.py

    def __init__(self, config, split):
        data = pd.read_csv("./data/gpt4_wiki_bios_full_evaluation_benchmark.csv")
        concept_path = './data/wiki_bio_concepts.txt'
        concepts = self.load_concepts(concept_path)
        concepts = [s.strip() for s in concepts]
        
        frac = int(config["data_split"]*len(concepts))
        edit_data = data[:frac].copy().reset_index()
        edit_concepts = concepts[:frac]        
        
        unlearn_data = data[frac:].copy().reset_index()
        unlearn_concepts = concepts[frac:]

        if split == "continued_pretraining":
            upstream_text = WebText10k(config, train_split=True, eval_split=False)
            wiki_bio_text = self.get_data(data, concepts, data_type="wiki_bio_text")
            self.text = wiki_bio_text + upstream_text.text
        elif split == "baseline_continued_pretraining":
            upstream_text = WebText10k(config, train_split=True, eval_split=False)
            edit_wiki_bio_text = self.get_data(edit_data, edit_concepts, data_type="gpt4_bio_text")
            self.text = edit_wiki_bio_text + upstream_text.text
        elif split == "unlearn":
            self.text = self.get_data(unlearn_data, unlearn_concepts, data_type="wiki_bio_text")
        elif split == "edit_unlearn":
            self.text = self.get_data(edit_data, edit_concepts, data_type="wiki_bio_text")
        elif split == "edit_update":
            self.text = self.get_data(edit_data, edit_concepts, data_type="gpt4_bio_text")
        elif split == "val_unlearn":
            self.text = self.get_eval_data(unlearn_data, eval_type="wiki_bio_text_qa", eval_split='validation')
        elif split == "val_edit_unlearn":
            self.text = self.get_eval_data(edit_data, eval_type="wiki_bio_text_qa", eval_split='validation')
        elif split == "val_edit_update":
            self.text = self.get_eval_data(edit_data, eval_type="gpt4_inaccurate_text_qa", eval_split='validation')
        elif split == "test_unlearn":
            self.text = self.get_eval_data(unlearn_data, eval_type="wiki_bio_text_qa", eval_split='test')
        elif split == "test_edit_unlearn":
            self.text = self.get_eval_data(edit_data, eval_type="wiki_bio_text_qa", eval_split='test')
        elif split == "test_edit_update":
            self.text = self.get_eval_data(edit_data, eval_type="gpt4_inaccurate_text_qa", eval_split='test')
            
    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]

    def load_concepts(self, PATH):
        with open(PATH,'r') as f:
            concepts = f.readlines()
        return concepts
    
    def get_data(self, data, concepts, data_type):
        _text = []
        for i in range(len(data)):
            header = f"This is a Wikipedia passage about {concepts[i]}."
            _text.append(f"{header} {data[data_type][i]}")  
        return [{"text": s, "labels": [], "concept": []} for s in _text]
    
    def replace_answer(self, qa):
        if not "option" in qa['correct answer']:
            if qa['correct answer'] == qa['option1']:
                qa['correct answer'] = 'option1'
            elif qa['correct answer'] == qa['option2']:
                qa['correct answer'] = 'option2'
            elif qa['correct answer'] == qa['option3']:
                qa['correct answer'] = 'option3'
        return qa
    
    def get_eval_data(self, data, eval_type, eval_split):
        eval = []
        for _, row in data.iterrows():
            try:
                qa_pairs = ast.literal_eval(row[eval_type])
            except:
                continue
            qa_pairs_val, qa_pairs_test = train_test_split(qa_pairs, shuffle=True, test_size=0.5, random_state=42)
            if eval_split == 'validation':
                for qa in qa_pairs_val:
                    if not "option" in qa['correct answer']:
                        qa = self.replace_answer(qa)
                    eval.append({"question":qa['Q'], "option1": qa['option1'], "option2": qa['option2'], "option3": qa['option3'], "labels": qa['correct answer']})
            elif eval_split == 'test':
                for qa in qa_pairs_test:
                    if not "option" in qa['correct answer']:
                        qa = self.replace_answer(qa)
                    eval.append({"question":qa['Q'], "option1": qa['option1'], "option2": qa['option2'], "option3": qa['option3'], "labels": qa['correct answer']})
        return eval

class ARCEasy(Dataset):
    def __init__(self, config, split):
        if split == 'validation':
            eval_data_arc_easy = pd.read_csv("./data/arc_easy_validation.csv")
        elif split == 'test':
            eval_data_arc_easy = pd.read_csv("./data/arc_easy_test.csv")
        self.text = []
        for idx , row in eval_data_arc_easy.iterrows():
            choices = ast.literal_eval(row['choices'])
            if len(choices['label']) == 4:
                correct_option = self.map_label_to_option(choices['label'].index(row['answerKey']))
                self.text.append({"question": row['question'],
                                  "option1": choices['text'][0], 
                                  "option2": choices['text'][1], 
                                  "option3": choices['text'][2], 
                                  "option4": choices['text'][3], 
                                  "labels": correct_option})

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]
    
    def map_label_to_option(self, label):
        if label == 0:
            return 'option1'
        elif label == 1:
            return 'option2'
        elif label == 2:
            return 'option3'
        elif label == 3:
            return 'option4'
        
'''
config = {}
config["data_path"] = "/fsx/sambits_interns_2024/data/"
config["data_split"] = 0.5
config["retain_train_split"] = 2079
config["retain_test_split"] = 2079
upstream_text = WebText10k(config, train_split=True, eval_split='validation')
#upstream_text = WikiBio(config, split="baseline_continued_pretraining")
#upstream_text = ARCEasy(config, split="validation")
for i in upstream_text.text[:10]:
    print(i)
    print("="*20)
print(len(upstream_text.text))
'''

def tokenize(batch, tokenizer, local_rank, test=True):
    if test:
        prompt, label = batch["text"], batch["labels"]
        mask_token = -100 # ignore_index of CrossEntropyLoss
        full_prompt = [f"{p} {l}" for p, l in zip(prompt, label)]
        prompt_ids = tokenizer(list(prompt), return_tensors="pt", padding=True)["input_ids"]
        num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in prompt_ids]
        tokens = tokenizer(full_prompt, return_tensors="pt", padding=True)
        tokens["labels"] = tokens["input_ids"].clone()
        for i in range(len(prompt)):
            tokens["labels"][i][:num_prompt_toks[i]] = mask_token
        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token 
    else:
        prompt, label = batch["text"], batch["labels"]
        mask_token = -100 # ignore_index of CrossEntropyLoss
        tokens = tokenizer(list(prompt), return_tensors="pt", padding=True)
        tokens["labels"] = tokens["input_ids"].clone()
        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token

    tokens = {f"{k1}" : v1.to(local_rank) for k1, v1 in tokens.items()}
    return tokens

def perplexity(model, tokenizer, loader, local_rank, test=True):
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    num_examples = 0
    total_ppl = 0
    with torch.no_grad():
        for _, batch in enumerate(iter(loader)):
            batch = tokenize(batch, tokenizer, local_rank, test=test)
            logits = model(**batch).logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            # Flatten the tokens
            #loss = loss_fct(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))
            #loss = loss.view(shift_labels.size(0), shift_labels.size(1))
            loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
            total_ppl += torch.sum(torch.tensor([torch.exp(torch.mean(loss[row][loss[row]!=0])) for row in range(loss.size(0))]))
            num_examples += len(batch["labels"])   
        total_ppl = total_ppl.to(local_rank)
        num_examples = torch.tensor(num_examples).to(local_rank)
    return total_ppl, num_examples

def accuracy(model, tokenizer, loader, local_rank, test=True):
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        options = ["option1", "option2", "option3"]
        correct = 0
        num_examples = 0
        for _, batch in enumerate(iter(loader)):
            ppls = []
            for option in options:
                temp_batch = {"text": batch["question"], "labels": batch[option]}
                temp_batch = tokenize(temp_batch, tokenizer, local_rank, test=True)
                logits = model(**temp_batch).logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = temp_batch["labels"][..., 1:].contiguous()
                # Flatten the tokens
                #loss = loss_fct(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))
                #loss = loss.view(shift_labels.size(0), shift_labels.size(1))
                loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
                temp_ppl = torch.tensor([torch.exp(torch.mean(loss[row][loss[row]!=0])) for row in range(loss.size(0))])
                ppls.append(temp_ppl.unsqueeze(0))
            ppls = torch.cat(ppls, dim=0)
            lowest_ppl_indices = torch.argmin(ppls, dim=0)
            gt_indices = torch.tensor([options.index(gt_option) for gt_option in batch["labels"]])
            correct += torch.sum(lowest_ppl_indices==gt_indices)
            num_examples += len(batch["labels"])
        correct = correct.to(local_rank)
        num_examples = torch.tensor(num_examples).to(local_rank)
    return correct, num_examples

def accuracy_arc_easy(model, tokenizer, loader, local_rank, test=True):
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        options = ["option1", "option2", "option3", "option4"]
        correct = 0
        num_examples = 0
        for _, batch in enumerate(iter(loader)):
            ppls = []
            for option in options:
                temp_batch = {"text": batch["question"], "labels": batch[option]}
                temp_batch = tokenize(temp_batch, tokenizer, local_rank, test=True)
                logits = model(**temp_batch).logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = temp_batch["labels"][..., 1:].contiguous()
                # Flatten the tokens
                #loss = loss_fct(shift_logits.view(-1, tokenizer.vocab_size), shift_labels.view(-1))
                #loss = loss.view(shift_labels.size(0), shift_labels.size(1))
                loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
                temp_ppl = torch.tensor([torch.exp(torch.mean(loss[row][loss[row]!=0])) for row in range(loss.size(0))])
                ppls.append(temp_ppl.unsqueeze(0))
            ppls = torch.cat(ppls, dim=0)
            lowest_ppl_indices = torch.argmin(ppls, dim=0)
            gt_indices = torch.tensor([options.index(gt_option) for gt_option in batch["labels"]])
            correct += torch.sum(lowest_ppl_indices==gt_indices)
            num_examples += len(batch["labels"])
        correct = correct.to(local_rank)
        num_examples = torch.tensor(num_examples).to(local_rank)
    return correct, num_examples

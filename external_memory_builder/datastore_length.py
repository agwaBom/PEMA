import torch
from random_seed import set_random_seed
from transformers import OPTModel, OPTConfig, AutoTokenizer, OPTForCausalLM, AutoConfig, AutoModelForCausalLM, DataCollatorWithPadding, AdamW, GPT2Tokenizer, DataCollatorForLanguageModeling, GenerationConfig
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, Accelerator
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from typing import List
import torch.nn.functional as F
import torch.nn as nn
import argparse
from tqdm import tqdm
import datasets
from datasets import inspect_dataset, load_dataset_builder, load_metric, load_dataset
import evaluate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_random_seed(123)
import numpy as np

def tokenize_input(examples):
    type = "wmt"
    if type == "wmt":
        inputs = ["Translate this from English to German: \n" + en + "\nGerman: " for en, du in zip(examples['en'], examples['du'])]
    elif type == "gyafc":
        inputs = ["Convert the following informal sentence into a formal sentence: \nInformal: " + informal + "\nFormal: " for informal in examples['informal']]

    # model_inputs outputs dict {"input_ids", "attention_mask"}
    model_inputs = tokenizer(inputs, max_length=500, add_special_tokens=True)
    return model_inputs

def tokenize_label(examples):
    type = "wmt"
    if type == "wmt":
        labels = ["Translate this from English to German: \n" + en + "\nGerman: " + du for en, du in zip(examples['en'], examples['du'])]
    elif type == "gyafc":
        labels = ["Convert the following informal sentence into a formal sentence: \nInformal: " + informal + "\nFormal: " + formal for informal, formal in zip(examples['informal'], examples['formal'])]

    # model_inputs outputs dict {"input_ids", "attention_mask"}
    model_labels = tokenizer(labels, max_length=500, add_special_tokens=True)
    return model_labels

def load_sharded_checkpoint(checkpoint, path):
    # https://huggingface.co/docs/accelerate/usage_guides/big_modeling
    config = AutoConfig.from_pretrained(checkpoint)
    with init_empty_weights():
        #model = OPTForCausalLM(config).half()
        model = AutoModelForCausalLM.from_config(config).half()
    model = load_checkpoint_and_dispatch(model, path, device_map="auto", no_split_module_classes=["OPTDecoderLayer"])
    return model

def load_non_sharded_checkpoint(checkpoint, device):
    model = OPTForCausalLM.from_pretrained(checkpoint)
    return model.to(device)

def save_pretrained_weight(path, model):
    model.save_pretrained(path)
    print(sorted(os.listdir(path)))
# 10 3445/4548
if __name__ == "__main__":
    # Basic Argument Setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20) # 약 4G 모델
    parser.add_argument("--model", default="facebook/opt-iml-max-1.3b")
    parser.add_argument("--data_path", default="wmt22/news_commentary_16_en_de_train_0.json")

    opt =  parser.parse_args()
    batch_size = opt.batch_size
    model_path = "./"+opt.model
    accelerator = Accelerator()

    ## Load OPT model and shard
    checkpoint = opt.model
    non_sharded_checkpoint_list = ["facebook/opt-125m", "facebook/opt-1.3b", "facebook/opt-iml-1.3b", "facebook/opt-iml-max-1.3b"]

    # load config
    generation_config = GenerationConfig.from_pretrained("./", _from_pipeline='text-generation')
    generation_config.output_hidden_states=True
    generation_config.return_dict_in_generate=True

    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint, padding_side='left', _from_pipeline='text-generation')

    # load model
    try:
        if checkpoint in non_sharded_checkpoint_list:
            model = load_non_sharded_checkpoint(checkpoint, device)
        else:
            model = load_sharded_checkpoint(checkpoint, model_path)
    except:
        print("Loading Local Model Failed. Downloading a New Model ..")
        model = OPTForCausalLM.from_pretrained(checkpoint)
        os.mkdir("./"+checkpoint)
        save_pretrained_weight("./"+checkpoint, model)
        if checkpoint in non_sharded_checkpoint_list:
            model = load_non_sharded_checkpoint(checkpoint, device)
        else:
            model = load_sharded_checkpoint(checkpoint, model_path)

    print("Model Loaded.")
    model.generation_config = generation_config

    dataset = load_dataset("json", data_files=opt.data_path)
    dataset = dataset['train']
    print(f"total dataset length: {len(dataset)}")

    # add data
    tokenized_input = dataset.map(tokenize_input, batched=True, num_proc=20)
    tokenized_label = dataset.map(tokenize_label, batched=True, num_proc=20)

    tokenized_input = tokenized_input.remove_columns(["en", "du"])
    tokenized_label = tokenized_label.remove_columns(["en", "du"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    test_dataloader_input = DataLoader(tokenized_input, batch_size=batch_size, num_workers=10, collate_fn=data_collator)
    test_dataloader_label = DataLoader(tokenized_label, batch_size=batch_size, num_workers=10, collate_fn=data_collator)

    model.eval()
    total_label_idx_length = 0
    print(len(test_dataloader_input))
    print("Validating...")
    for i, (batch, batch_label) in tqdm(enumerate(zip(test_dataloader_input, test_dataloader_label))):
        print(f"========================BATCH: {i}=========================\n")
        batch = {k: torch.Tensor(v).to(device) for k, v in batch.items()}
        batch_label = {k: v.to(device) for k, v in batch_label.items()}
        print(f"input_ids: {batch['input_ids'].shape}")
        print(f"atten_mask: {batch['attention_mask'].shape}")

        cnt_pad_input = (batch['input_ids'] == 1.).sum(dim=-1)
        cnt_pad_label = (batch_label['input_ids'] == 1.).sum(dim=-1)

        # batch_size
        for b in range(batch['input_ids'].shape[0]):
            label_idxs = batch_label['input_ids'][b][cnt_pad_label[b]:][len(batch['input_ids'][b][cnt_pad_input[b]:])-1:]
            total_label_idx_length += len(label_idxs)

    print("total length: ", total_label_idx_length)
    
import torch
from random_seed import set_random_seed
from transformers import OPTModel, OPTConfig, AutoTokenizer, OPTForCausalLM, AutoConfig, AutoModelForCausalLM, DataCollatorWithPadding, AdamW, GPT2Tokenizer, DataCollatorForLanguageModeling, GenerationConfig
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, Accelerator
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import torch.nn as nn
import argparse
from tqdm import tqdm
import datasets
from datasets import inspect_dataset, load_dataset_builder, load_metric, load_dataset
import evaluate
import os
import pickle
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
        model = AutoModelForCausalLM.from_config(config).half()
    model = load_checkpoint_and_dispatch(model, path, device_map="auto", no_split_module_classes=["OPTDecoderLayer"])
    return model

def load_non_sharded_checkpoint(checkpoint, device):
    model = OPTForCausalLM.from_pretrained(checkpoint)
    return model.to(device)

def save_pretrained_weight(path, model):
    model.save_pretrained(path)
    print(sorted(os.listdir(path)))

if __name__ == "__main__":
    # Basic Argument Setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=30, type=int) # 약 4G 모델
    parser.add_argument("--model", default="facebook/opt-iml-max-1.3b")
    parser.add_argument("--em_size", default=609262, type=int)
    parser.add_argument("--data_path", default="wmt22/news_commentary_16_en_de_train_0.json")
    # dimension of the context representation
    # opt-125m: 768, 1.3b: 2048, 6.7b: 4096, 13b:5120, 30b: 7168
    parser.add_argument("--decoder_embed_dim", default=2048, type=int) 
    parser.add_argument("--dstore_fp16", default=False)
    parser.add_argument("--dstore_mmap", default="")
    parser.add_argument("--knn_interpolation_bool", default=False, action='store_true')
    parser.add_argument('--lora_trained', default=False, action='store_true', help='use path for lora')

    opt =  parser.parse_args()
    batch_size = opt.batch_size
    model_path = "./"+opt.model
    opt.dstore_mmap = f"./dstore"
    print(opt)

    accelerator = Accelerator()

    ## Load OPT model and shard
    checkpoint = opt.model
    non_sharded_checkpoint_list = ["facebook/opt-125m", "facebook/opt-1.3b", "facebook/opt-iml-1.3b", "facebook/opt-iml-max-1.3b"]

    generation_config = GenerationConfig.from_pretrained("./", config_file_name ='generation_config.json', _from_pipeline='text-generation')
    generation_config.output_hidden_states=True
    generation_config.return_dict_in_generate=True
    generation_config.output_scores=True
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint, padding_side='left', _from_pipeline='text-generation')

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

    tokenized_input = dataset.map(tokenize_input, batched=True, num_proc=10)
    tokenized_label = dataset.map(tokenize_label, batched=True, num_proc=10)

    tokenized_input = tokenized_input.remove_columns(["en", "du"])
    tokenized_label = tokenized_label.remove_columns(["en", "du"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    test_dataloader_input = DataLoader(tokenized_input, batch_size=batch_size, num_workers=10, collate_fn=data_collator)
    test_dataloader_label = DataLoader(tokenized_label, batch_size=batch_size, num_workers=10, collate_fn=data_collator)

    max_seq_len = max([len(i) for i in tokenized_input['input_ids']]) + 30 # 30 is for giving some space for the model to generate

    model.eval()
    total_pred_list = list()
    total_label_list = list()
    total_input_list = list()

    # Initialize external memory
    if opt.dstore_fp16:
        print('Saving fp16')
        dstore_keys = np.memmap(opt.dstore_mmap+'_keys.npy', dtype=np.float16, mode='w+', shape=(opt.em_size, opt.decoder_embed_dim))
        dstore_vals = np.memmap(opt.dstore_mmap+'_vals.npy', dtype=np.int16, mode='w+', shape=(opt.em_size, 1))
    else:
        print('Saving fp32')
        dstore_keys = np.memmap(opt.dstore_mmap+'_keys.npy', dtype=np.float32, mode='w+', shape=(opt.em_size, opt.decoder_embed_dim))
        dstore_vals = np.memmap(opt.dstore_mmap+'_vals.npy', dtype=np.int32, mode='w+', shape=(opt.em_size, 1))

        pt_dstore_keys = torch.empty((opt.em_size, opt.decoder_embed_dim), dtype=torch.float32)
        pt_dstore_vals = torch.empty((opt.em_size, 1), dtype=torch.int32)

    total_label_idx_length = 0
    dataloader_len = len(test_dataloader_input)
    print(dataloader_len)
    print("Validating...")
    for i, (batch, batch_label) in tqdm(enumerate(zip(test_dataloader_input, test_dataloader_label))):
        print(f"========================BATCH: {i}=========================\n")
        print(f"========================BATCH/LENGTH: {i}/{dataloader_len} =========================\n")
        batch = {k: torch.Tensor(v).to(device) for k, v in batch.items()}
        batch_label = {k: v.to(device) for k, v in batch_label.items()}
        print(f"input_ids: {batch['input_ids'].shape}")
        print(f"atten_mask: {batch['attention_mask'].shape}")

        max_seq_len = batch_label['input_ids'].shape[-1] + 10 # 10 is for giving some space for the model to generate
        print("Max Sequence: ",max_seq_len)
        
        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], max_length=max_seq_len, generation_config=generation_config, opt_args=opt)

        cnt_pad_input = (batch['input_ids'] == 1.).sum(dim=-1)
        cnt_pad_label = (batch_label['input_ids'] == 1.).sum(dim=-1)

        # batch_size
        for b in range(batch['input_ids'].shape[0]):
            label_idxs = batch_label['input_ids'][b][cnt_pad_label[b]:][len(batch['input_ids'][b][cnt_pad_input[b]:])-1:]
            # 0 for fn hid, 1 for last hidden
            # b for current index
            label_hidden = [hs[0][b] for hs in outputs.hidden_states[:len(label_idxs)]]

            # First hidden feature of the output can contain whole hidden feature of input, then only use their last hidden feature
            if label_hidden[0].shape[0] > 1:
                label_hidden[0] = label_hidden[0][-1].unsqueeze(dim=0)

            label_hidden = torch.stack(label_hidden, dim=0).squeeze()

            # if only processing 1 word, torch.stack(label_hidden, dim=0).squeeze() will squeeze [1, 768] to [768] which is not what we want
            if len(label_idxs) == 1:
                label_hidden = label_hidden.unsqueeze(dim=0)

            if len(label_idxs) != len(label_hidden):
                if len(label_idxs) > len(label_hidden):
                    label_idxs = label_idxs[:len(label_hidden)]
                else:
                    import IPython; IPython.embed()
            try:
                if opt.dstore_fp16:
                    # loop for examples and save [target_len, hidden_dim]
                    dstore_keys[total_label_idx_length:len(label_idxs)+total_label_idx_length] = label_hidden.cpu().numpy().astype(np.float16)
                    dstore_vals[total_label_idx_length:len(label_idxs)+total_label_idx_length] = label_idxs.view(-1, 1).cpu().numpy().astype(np.int16)
                else:
                    dstore_keys[total_label_idx_length:len(label_idxs)+total_label_idx_length] = label_hidden.cpu().numpy().astype(np.float32)
                    dstore_vals[total_label_idx_length:len(label_idxs)+total_label_idx_length] = label_idxs.view(-1, 1).cpu().numpy().astype(np.int32)
            except:
                print("dstore problem")
                import IPython; IPython.embed()

            pt_dstore_keys[total_label_idx_length:len(label_idxs)+total_label_idx_length] = label_hidden.cpu()
            pt_dstore_vals[total_label_idx_length:len(label_idxs)+total_label_idx_length] = label_idxs.view(-1, 1).cpu()

            total_label_idx_length += len(label_idxs)
        
    print("total length: ", total_label_idx_length)
    # Save pickle in case of memory error
    with open(opt.dstore_mmap+'_keys.pickle', 'wb') as f:
        pickle.dump(pt_dstore_keys[:total_label_idx_length], f, pickle.HIGHEST_PROTOCOL)

    with open(opt.dstore_mmap+'_vals.pickle', 'wb') as f:
        pickle.dump(pt_dstore_vals[:total_label_idx_length], f, pickle.HIGHEST_PROTOCOL)
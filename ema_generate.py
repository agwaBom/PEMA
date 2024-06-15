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
from datasets import load_dataset
import evaluate
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_random_seed(123)

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
        model = AutoModelForCausalLM.from_config(config).half() # half precision
    #"balanced_low_0"
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
    parser.add_argument("--model", default="facebook/opt-iml-max-30b")
    parser.add_argument("--log_dir", default="./runs/opt-iml-max-30b")
    parser.add_argument("--data_path", default="./datastore_builder/wmt/test.json")
    parser.add_argument("--output_dir", default="./output_result/en-de_11276358_train_1_3b/")

    parser.add_argument("--head_path", default="./opt-last-layer/opt_iml_max_1_3b_lm_head.pt")
    parser.add_argument("--lm_norm_path", default="./opt-last-layer/opt_iml_max_1_3b_layer_norm.pt")

    # KNN Argument Setting
    parser.add_argument("--interpolation_bool", default=False, action='store_true')
    parser.add_argument("--decoder_embed_dim", default=2048, type=int)
    parser.add_argument("--lmbda", default=0.0, type=float)
    parser.add_argument('--probe', default=32, type=int, help='for FAISS, the number of lists to query')
    # originally k==1024
    parser.add_argument('--k', default=1024, type=int, help='number of nearest neighbors to retrieve')
    parser.add_argument('--dstore_size', default=11276358, type=int, help='number of items in the knnlm datastore')
    parser.add_argument('--dstore_filename', type=str, default='./datastore/gyafc_1386996_train_30b_Combined/method_1/dstore', help='File where the knnlm datastore is saved')
    parser.add_argument('--indexfile', type=str, default='./datastore/gyafc_1386996_train_30b_Combined/method_1/knn.index', help='File containing the index built using faiss for knn')
    parser.add_argument('--faiss_metric_type', default='l2', type=str, help='the distance metric for faiss')
    #parser.add_argument('--dstore-fp16', default=False, action='store_true', help='if true, datastore items are saved in fp16 and int16')
    parser.add_argument('--dstore_fp16', default=False, action='store_true', help='if true, datastore items are saved in fp16 and int16')
    parser.add_argument('--move_dstore_to_mem', default=False, action='store_true', help='move the keys and values for knn to memory')
    parser.add_argument('--fp16', default=False, action='store_true', help='use FP16')
    parser.add_argument('--knn_sim_func', default=None, type=str, help='similarity function to use for knns')
    parser.add_argument('--no_load_keys', default=False, action='store_true', help='do not load keys')
    parser.add_argument('--lora_decoder', default=False, action='store_true', help='use lora decoder')

    parser.add_argument('--gradual_unrolling', default=False, action='store_true', help='gradually unroll the interpolation')

    # min: 1.0 (same as not applying), max: inf
    # 1.11111==10%
    # 1.33333==25%
    # 2.0==50%
    # 4.0==75%
    parser.add_argument('--gradual_unrolling_min', default=0, type=float, help='gradually unroll the interpolation (percentage %)')

    # LORA Argument Setting
    parser.add_argument('--num_rank', default=8, type=int, help='number of ranks to use for distributed training')
    parser.add_argument('--lora_linear_path', type=str, default='./datastore/gyafc_1386996_train_30b_Combined/method_1/lora_task.pt', help='use path for lora')
    parser.add_argument('--lora_trained', default=False, action='store_true', help='use path for lora')
    opt =  parser.parse_args()

    print(opt)

    # Calculate Gradual Unrolling min value. However, we did not apply min value in the PEMA paper.
    print("values_percentage:", opt.gradual_unrolling_min)
    y = 1/(1-opt.gradual_unrolling_min*0.01)
    opt.gradual_unrolling_min = y
    print("values_changed:", opt.gradual_unrolling_min)

    writer = SummaryWriter(opt.log_dir)
    batch_size = opt.batch_size
    model_path = "./model/"+opt.model
    os.mkdir(model_path)
    writer.add_text('model', opt.model)
    writer.add_text('batch_size', str(batch_size))
    accelerator = Accelerator()

    ## Load OPT model and shard
    non_sharded_checkpoint_list = ["facebook/opt-125m", "facebook/opt-1.3b", "facebook/opt-iml-1.3b", "facebook/opt-iml-max-1.3b", "model_weight/facebook/opt-iml-max-1.3b_loss_7.46_step_1000"]

    generation_config = GenerationConfig.from_pretrained("./", config_file_name ='generation_config.json', _from_pipeline='text-generation')
    generation_config.output_hidden_states=True
    generation_config.return_dict_in_generate=True
    generation_config.output_scores=True
    tokenizer = GPT2Tokenizer.from_pretrained(opt.model, padding_side='left', _from_pipeline='text-generation')
    label_tokenizer = GPT2Tokenizer.from_pretrained(opt.model, padding_side='right', _from_pipeline='text-generation')

    try:
        if opt.model in non_sharded_checkpoint_list:
            model = load_non_sharded_checkpoint(opt.model, device)
        else:
            model = load_sharded_checkpoint(opt.model, model_path)
    except:
        print("Loading Local Model Failed. Downloading a New Model ..")
        model = OPTForCausalLM.from_pretrained(opt.model)
        os.mkdir(model_path)
        save_pretrained_weight(model_path, model)
        if opt.model in non_sharded_checkpoint_list:
            model = load_non_sharded_checkpoint(opt.model, device)
        else:
            model = load_sharded_checkpoint(opt.model, model_path)

    print("Model Loaded.")
    model.generation_config = generation_config

    # Load Dataset of informal input and formal.ref0 label
    dataset = load_dataset("json", data_files=opt.data_path)
    dataset = dataset["train"]

    print(f"total dataset length: {len(dataset)}")
    # add data
    tokenized_input = dataset.map(tokenize_input, batched=True, num_proc=10)
    tokenized_label = dataset.map(tokenize_label, batched=True, num_proc=10)
    
    tokenized_input = tokenized_input.remove_columns(["en", "du"])
    tokenized_label = tokenized_label.remove_columns(["en", "du"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    data_collator_label = DataCollatorForLanguageModeling(tokenizer=label_tokenizer, mlm=False)

    test_dataloader_input = DataLoader(tokenized_input, batch_size=batch_size, num_workers=10, collate_fn=data_collator)
    test_dataloader_label = DataLoader(tokenized_label, batch_size=batch_size, num_workers=10, collate_fn=data_collator_label)

    bleu = evaluate.load("bleu")
    model.eval()
    total_pred_list = list()
    total_label_list = list()
    total_input_list = list()
    
    total_label_idx_length = 0
    print("Validating...")
    for i, (batch, batch_label) in enumerate(tqdm(zip(test_dataloader_input, test_dataloader_label))):
        print(f"========================BATCH/LENGTH: {i}/{len(test_dataloader_input)} =========================\n")
        batch = {k: torch.Tensor(v).to(device) for k, v in batch.items()}
        batch_label = {k: v.to(device) for k, v in batch_label.items()}
        #print(f"input_ids: {batch['input_ids'].shape}")
        #print(f"atten_mask: {batch['attention_mask'].shape}")
        
        max_seq_len = batch_label['input_ids'].shape[-1] + 10 # 10 is for giving some space for the model to generate
        print("Max Sequence: ",max_seq_len)

        with torch.no_grad():
            outputs = model.generate(input_ids=batch['input_ids'], 
                                     attention_mask=batch['attention_mask'], 
                                     max_length=max_seq_len, 
                                     generation_config=generation_config, 
                                     label_ids=batch_label['input_ids'][:, 1:],
                                     opt_args=opt)
                                     #do_sample=True, 
                                     #num_beams=5,
                                     #renormalize_logits=True,
                                     #early_stopping=True)

            input_list = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            pred_list = tokenizer.batch_decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            label_list = tokenizer.batch_decode(batch_label['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        total_pred_list.append(pred_list)
        total_label_list.append(label_list)
        total_input_list.append(input_list)

        print("##LABLE##")
        print(label_list[0])
        print("##PREDICTION##")
        print(pred_list[0].replace(input_list[0], ""))

    total_label_list_tmp = [j for i in total_label_list for j in i]
    total_pred_list_tmp = [j for i in total_pred_list for j in i]
    total_input_list_tmp = [j for i in total_input_list for j in i]
    
    refined_pred_list = list()
    for i, p in zip(total_input_list_tmp, total_pred_list_tmp):
        refined_pred_list.append(p.replace(i, ""))
    
    refined_pred_list = [i.replace('\x03', '').replace('\r', '').replace('\x13', '') for i in refined_pred_list]

    # make output dir if not exist
    if opt.output_dir is not None:
        # exist_ok=True: if the directory already exists, do not raise an error
        os.makedirs(opt.output_dir, exist_ok=True)

    with open(opt.output_dir+'/label.txt', mode='w') as out:
        for i in total_label_list_tmp:
            out.write(i+'\n')

    with open(opt.output_dir+'/lm_pred.txt', mode='w') as out:
        for i in refined_pred_list:
            out.write(i.replace("\n", "")+'\n')

    with open(opt.output_dir+'/input.txt', mode='w') as out:
        for i in total_input_list_tmp:
            out.write(i+'\n')
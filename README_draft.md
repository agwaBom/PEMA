# PEMA: An Offsite-Tunable Plug-in External Memory Adaptation for Language Models [[paper](https://arxiv.org/abs/2311.08590)]

<p align="center">
<img src="figures/overview.png" width="600">
</p>
<p align = "center">
Figure 1. A motivation for PEMA. (a) The data owners who want to fine-tune PLMs encounter a problem when the PLM owner refuses to share all the weights of the PLM. (b) In the PEMA training phase, the data owner takes a CR from the PLM owner by providing a context prompt. They subsequently train their PEMA model with their dataset. (c) At inference, the data owner takes a CR for test data from the PLM owner. Using Gradual Unrolling (GU), they generate the next-token by interpolating between PEMA and PLM next-token probabilities.
</p>

This is the official implementation of the paper "PEMA: An Offsite-Tunable Plug-in External Memory Adaptation for Language Models". Which is accepted at 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics [NAACL 2024](https://2024.naacl.org/).

## Requirements
- We use three RTX 8000 GPUs with 48GB GDDR6 memory for our experiment. 
- For OPT-IML-MAX1.3B, we use full precision (FP32) for training and inference. For LLaMA-7B and OPT-IML-MAX30B, we use half-precision (FP16) and distribute the model across three GPUs using the HuggingFace Accelerate library.

## Installation

1. Download & install the latest anaconda distribution
```
# Download anaconda distribution for Ubuntu
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh

# installation
sh Anaconda3-2023.07-2-Linux-x86_64.sh
```

2. Once installation is complete, create a new environment named "pema_opt"
```
# Created new conda environment
conda create --name pema_opt

# Activate enviornment
conda activate pema_opt
```

3. Install required library with conda env update. However, huggingface and accelerate libraries should installed separately.
```
# Install requrired library
conda env update --name pema_opt --file environment_pema_opt.yml 

# Install huggingface
cd transformers
pip install -e .
cd ..

# Install accelerate
cd accelerate
pip install -e .
cd ..
```

## Download Full Dataset (Can be skipped)
- Due to limited budget on uploading size, we use a subset of wmt22 dataset for demonstration. However, full dataset can be found below.
```
# GYAFC can be accessed via requesting corpus in email
# For the detail, please refer this repo
https://github.com/raosudha89/GYAFC-corpus

# WMT22 de-en can be downloaded here
https://data.statmt.org/news-commentary/v16/training/

# We use WMT22 test set which is in here
https://github.com/microsoft/gpt-MT

# Downloaded parallel corpus should be transformed into single .json format
python parallel_corpus_to_json.py \
    --path [path to dataset] \
    --en_path [path to english corpus] \
    --du_path [path to german corpus]
```

## Architecture
<p align="center">
<img src="figures/architecture.png" width="900">
</p>
<p align = "center">
Figure 2. An illustration of PEMA. The areas of the PLM owner and the data owner are separated by the blue horizontal line. The data owner can train and infer using only the PLM's LM head. PEMA builds an external memory from the training context with an instruction $[Inst]$ given to a PLM. The PLM outputs the representation $f(c_i)$ and predicts the next-token distribution $P_{LM}(\hat{w}_i)$. The representation $f(c_i)$ is then aligned with its target $y_i$. In the training phase, PEMA uses external memory for two tasks: preserving the original representation via reconstruction training with $B_{rct}$ and generating a target token probability distribution using $B_{pd}$. For inference, the model inputs a test data representation to generate two probability distributions: $P_{LM}(\hat{w}_i)$ and $P_{PEMA}(\hat{w}_i)$. These are then interpolated using Gradual Unrolling to obtain the final token distribution.
</p>


## Building External Memory
To build an external memory, we first need how many context representation is needed to be saved.

It is because we use np.memmap to load from disk every time we load context representation.

To estimate the length of context representation run following code.
```
# Estimate total length (609,262)
python external_memory_builder/datastore_length.py
```

After retriving the length of context representation, we save the external memory by
```
python external_memory_builder/datastore_save.py \
    --em_size 609262 \
    --data_path wmt22/news_commentary_16_en_de_train_0.json \
    --model facebook/opt-iml-max-1.3b \
    --decoder_embed_dim 2048 \
```
Where 
- em_size = length of context representation
- data_path = data to build external memory
- model = what OPT model variant intend to use
- decoder_embed_dim = context representation size (hidden dimension of a model)
  - opt-125m: 768, 1.3b: 2048, 6.7b: 4096, 13b:5120, 30b: 7168

- After running the code will generate final_length of context representation (609,011)

## Train PEMA
Before trainin PEMA, we need last lm_head and layer_norm weight. To retrieve, use the command below. The weight will be saved in "./opt_last_layer" folder.
```
python external_memory_builder/save_lm_head_layer_norm.py
```

- When training PEMA, you can simple run "sh rct_pd_training.sh"
```
# Train rct (reconstruction decoder)
# Training script for RCT
CUDA_VISIBLE_DEVICES=0 python train_rct_pd.py  \
    --method 1 \
    --train_key_path "./dstore_keys.npy" \
    --train_val_path "./dstore_vals.npy" \
    --valid_key_path "./dstore_dev_keys.npy" \
    --valid_val_path "./dstore_dev_vals.npy" \
    --rct_path "" \
    --tensorboard_path "./runs/wmt22_rct_1_3b" \
    --train_dstore_size 609011 \
    --valid_dstore_size 202549 \
    --dstore_dim 2048 \
    --vocab_size 50272 \
    --save_path "./rct" \
    --num_rank 512 \
    --num_epochs 200 \
    --kappa 0.2 \
    --batch_size 20480


# Train rct_pd (joint retraining)
CUDA_VISIBLE_DEVICES=0 python train_rct_pd.py  \
    --method 3 \
    --train_key_path "./dstore_keys.npy" \
    --train_val_path "./dstore_vals.npy" \
    --valid_key_path "./dstore_dev_keys.npy" \
    --valid_val_path "./dstore_dev_vals.npy" \
    --tensorboard_path "./runs/wmt_rct_cls_1_3b_0" \
    --train_dstore_size 609011 \
    --valid_dstore_size 202549 \
    --dstore_dim 2048 \
    --vocab_size 50272 \
    --save_path "./rct_pd" \
    --num_rank 512 \
    --num_epochs 200 \
    --kappa 0.2 \
    --batch_size 2048 \
    --rct_path [trained_pema_rct_path]
```
Where 
- method: 1) train reconstruction 2) train classification 3) train joint retraining. (we only need 1 and 3 to train PEMA)
- train_key_path, train_val_path, valid_key_path, valid_val_path is external memory files (.npy)
- train_dstore_size: external memory size at training
- valid_dstore_size: external memory size at validation
- dstore_dim: context representation dimension (i.e., hidden size of a model)
- num_rank: number of rank size
- kappa: kappa value
- rct_path: path of pre-trained rct 

## Inference
Using a pretrained rct_pd (i.e., PEMA) we now can inference!

the core implementation code can be found at
"transformers/src/transformers/generation/utils.py"
- Used for interpolation between PEMA and PLM ouput
- Gradual Unrolling strategy

"transformers/src/transformers/models/opt/modeling_opt.py"
- Output context representation

Inference can be easily done with following code
```
# 90-0 Gradual Unrolling Interpolation
CUDA_VISIBLE_DEVICES=0 python ./ema_generate.py \
    --batch_size 30 \
    --model facebook/opt-iml-max-1.3b \
    --data_path wmt22/test.json \
    --output_dir ./pema_90_0 \
    --head_path ./opt_last_layer/opt_iml_max_1_3b_lm_head.pt \
    --lm_norm_path ./opt_last_layer/opt_iml_max_1_3b_layer_norm.pt \
    --interpolation_bool \
    --decoder_embed_dim 2048 \
    --lmbda 0.9 \
    --lora_decoder \
    --gradual_unrolling \
    --gradual_unrolling_min 0 \
    --num_rank 512 \
    --lora_linear_path wmt22_confirmed/rct_cls_0.2_lambda_5.311_loss.pt \
    --lora_trained
```

## Post-processing
Predicted setnece may convey hallucination.
To remove common hallucination occuring in sentence generation process,
we use "postprocess_generated_opt.py" for post-processing.
However, if there are additional hallucination patterns that are noticed, we also removed them 
```
python postprocess_generated_opt.py \
    --input ./pema_90_0/lm_pred.txt \
    --output ./pema_90_0/lm_pred_post.txt
```
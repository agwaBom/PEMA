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

# Training script for RCT_PD
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
    --rct_path "./wmt22_confirmed/rct_cls_0.2_lambda_5.311_loss.pt"

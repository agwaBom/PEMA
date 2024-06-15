import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import numpy as np
from torch.utils.data import Dataset
import copy
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from random_seed import set_random_seed
set_random_seed(123)

class LoraLinear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)

        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        if self.r > 0:
            #result = F.linear(x, T(self.weight), bias=self.bias)
            # x.shape, self.lora_dropout(x) [10, 10]
            # self.lora_A.T.shape [10, 5]
            # self.lora_B.T.shape [5, 10]
            result = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            r_result = (self.lora_dropout(x) @ self.lora_A.T) * self.scaling
            return result, r_result
        else:
            AssertionError('r should be greater than 0')

def train_rct(opt, lora_task_1, optimizer_1, train_dataloader, test_dataloader, device):
    loss_prev = 10000
    loss_prev_2 = 10000
    valid_cnt = 0
    print('train rct')

    # Method 1: Use lora_linear as a auto-encoder model
    early_stop = False
    for epoch in range(opt.num_epochs):
        # train step
        lora_task_1.train()
        loss_sum = 0
        for i, batch in enumerate(train_dataloader):
            current_step = epoch * len(train_dataloader) + i
            optimizer_1.zero_grad()
            features, labels = batch
            # Forward pass: Compute predicted y by passing x to the model
            y_pred, r_y_pred = lora_task_1(features.to(device))

            # Compute and print loss
            loss = F.mse_loss(y_pred.reshape(-1), features.reshape(-1).to(device))
            # Zero gradients, perform a backward pass, and update the weights.
            loss_sum += loss.item()

            print(f"Current/Total epoch step: {current_step}/{len(train_dataloader)}\tTrain step loss: {loss.item()}")
            writer.add_scalar('rct train step loss', loss.item(), current_step)
            loss.backward()
            optimizer_1.step()

            if current_step % 100 == 0:
                valid_cnt += 1
                total_valid_loss = 0
                lora_task_1.eval()
                for batch in tqdm(test_dataloader):
                    with torch.no_grad():
                        features, labels = batch
                        y_pred, r_y_pred = lora_task_1(features.to(device))
                        loss = F.mse_loss(y_pred.reshape(-1), features.reshape(-1).to(device))
                        total_valid_loss += loss.item()

                print('Test_loss:', total_valid_loss/len(test_dataloader))
                writer.add_scalar('rct_cls test loss', total_valid_loss/len(test_dataloader), valid_cnt)
                torch.save(lora_task_1.state_dict(), opt.save_path+'/rct_'+str(current_step)+'_step_'+str(round(total_valid_loss/len(test_dataloader), 3))+'_loss.pt')

                # give up if the loss is increasing for 2 consecutive validation steps
                if current_step > 1 and total_valid_loss > loss_prev and total_valid_loss > loss_prev_2:
                    print('Early stopping')
                    early_stop = True
                    break
                loss_prev_2 = loss_prev
                loss_prev = total_valid_loss
            if early_stop:
                break
        if early_stop:
            break
        print('Epoch:', epoch, '\nTrain_loss:', loss_sum / len(train_dataloader))


def train_rct_classification(opt, lora_task_3, lora_task_1, lm_head, final_layer_norm, optimizer_3, train_dataloader, test_dataloader, device):
    loss_prev = 10000
    loss_prev_2 = 10000
    valid_cnt = 0
    print('rct_classification')

    lora_task_1.eval()
    early_stop = False
    for epoch in range(opt.num_epochs):
        # train step
        lora_task_3.module.train()
        loss_sum = 0
        for i, batch in enumerate(train_dataloader):
            current_step = epoch * len(train_dataloader) + i
            optimizer_3.zero_grad()
            features, labels = batch
            y_dec_pred, r_y_pred = lora_task_3(features.to(device))

            # Compute reconstruction loss
            y_ae_pred = r_y_pred @ lora_task_1.module.lora_B.T
            loss_1 = F.mse_loss(y_ae_pred.reshape(-1), features.reshape(-1).to(device))

            # Compute word classification loss
            lm_output = lm_head(final_layer_norm(y_dec_pred))
            loss_2 = F.cross_entropy(lm_output, labels.reshape(-1).to(torch.long).to(device))

            # interpolate the loss
            total_loss = opt.kappa*loss_1 + (1-opt.kappa)*loss_2
            loss_sum += total_loss.item()
            print(f"Current/Total epoch step: {current_step}/{len(train_dataloader)}\tTrain step loss: {total_loss.item()}")
            writer.add_scalar('rct_cls train step loss', total_loss.item(), current_step)

            total_loss.backward()
            optimizer_3.step()

            if current_step % 100 == 0 and current_step != 0:
                valid_cnt += 1
                total_valid_loss = 0
                lora_task_3.module.eval()
                for batch in tqdm(test_dataloader):
                    with torch.no_grad():
                        features, labels = batch
                        y_dec_pred, r_y_pred = lora_task_3(features.to(device))

                        # Compute reconstruction loss
                        y_ae_pred = r_y_pred @ lora_task_1.module.lora_B.T
                        loss_1 = F.mse_loss(y_ae_pred.reshape(-1), features.reshape(-1).to(device))

                        # Compute word classification loss
                        lm_output = lm_head(final_layer_norm(y_dec_pred))
                        loss_2 = F.cross_entropy(lm_output, labels.reshape(-1).to(torch.long).to(device))

                        # interpolate the loss
                        total_loss = opt.kappa*loss_1 + (1-opt.kappa)*loss_2
                        total_valid_loss += total_loss.item()
                print('Test_loss:', total_valid_loss/len(test_dataloader))
                writer.add_scalar('rct_cls test loss', total_valid_loss/len(test_dataloader), valid_cnt)
                torch.save(lora_task_3.module.state_dict(), opt.save_path+'/rct_cls_'+str(current_step)+'_step_'+str(round(total_valid_loss/len(test_dataloader), 3))+'_loss.pt')
                # give up if the loss is increasing for 2 consecutive validation steps
                if current_step > 1 and total_valid_loss > loss_prev and total_valid_loss > loss_prev_2:
                    print('Early stopping')
                    early_stop = True
                    break
                loss_prev_2 = loss_prev
                loss_prev = total_valid_loss
            if early_stop:
                break
        if early_stop:
            break
        print('Epoch:', epoch, '\nTrain_loss:', loss_sum / len(train_dataloader))


def train_classification(opt, lora_task_2, lm_head, final_layer_norm, optimizer_2, train_dataloader, test_dataloader, device):
    # loss_prev = 10000

    for epoch in range(opt.num_epochs):
        # train step
        lora_task_2.module.train()
        loss_sum = 0
        for batch in tqdm(train_dataloader):
            optimizer_2.zero_grad()
            features, labels = batch
            # Forward pass: Compute predicted y by passing x to the model
            y_pred, r_y_pred = lora_task_2(features.to(device))
            lm_output = lm_head(final_layer_norm(y_pred))
                
            # Compute and print loss
            loss = F.cross_entropy(lm_output, labels.reshape(-1).to(torch.long).to(device))
            loss_sum += loss.item()

            loss.backward()
            optimizer_2.step()
        print('Epoch:', epoch, '\nTrain_loss:', loss_sum / len(train_dataloader))
        writer.add_scalar('cls train loss', loss_sum / len(train_dataloader), epoch)

        # validation step
        lora_task_2.module.eval()
        loss_sum = 0
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                features, labels = batch
                y_pred, r_y_pred = lora_task_2(features.to(device))

                # Forward pass: Compute predicted y by passing x to the model
                lm_output = lm_head(final_layer_norm(y_pred))
                #lm_pred = lm_output.argmax(dim=-1)
                
                # Compute and print loss
                loss = F.cross_entropy(lm_output, labels.reshape(-1).to(torch.long).to(device))
                loss_sum += loss.item()

        print('Test_loss:', loss_sum / len(test_dataloader))
        writer.add_scalar('cls test loss', loss_sum/len(test_dataloader), epoch)
        torch.save(lora_task_2.module.state_dict(), opt.save_path+str(round(loss_sum/len(test_dataloader), 3))+'_loss.pt')


def load_datastore(dstore_size, dstore_dim, key_path, val_path, batch_size=40960):
    # Load the datastore
    dstore_keys = np.memmap(key_path, dtype=np.float32, mode='r', shape=(dstore_size, dstore_dim))
    dstore_vals = np.memmap(val_path, dtype=np.int32, mode='r', shape=(dstore_size, 1))
    print('Datastore Loaded')

    batch_size = batch_size
    num_batches = (dstore_keys.shape[0] // batch_size)+1

    batched_dstore_keys = []
    batched_dstore_vals = []

    # Split the datastore into batches
    for i in tqdm(range(0, num_batches)):
        if i == num_batches:
            dstore_keys_cpy = dstore_keys[i*batch_size:]
            dstore_vals_cpy = dstore_vals[i*batch_size:]
            break
        dstore_keys_cpy = dstore_keys[i*batch_size:(i+1)*batch_size]
        dstore_vals_cpy = dstore_vals[i*batch_size:(i+1)*batch_size]

        batched_dstore_keys.append(torch.Tensor(dstore_keys_cpy))
        batched_dstore_vals.append(torch.Tensor(dstore_vals_cpy).to(torch.int32).reshape(-1))
    
    return batched_dstore_keys, batched_dstore_vals

class CustomDataset(Dataset):
    def __init__(self, key_path, val_path, dstore_size, dstore_dim):
        self.key_path = key_path
        self.val_path = val_path
        self.dstore_size = dstore_size
        self.dstore_dim = dstore_dim
        self.dstore_keys = np.memmap(self.key_path, dtype=np.float32, mode='r', shape=(self.dstore_size, self.dstore_dim))
        self.dstore_vals = np.memmap(self.val_path, dtype=np.int32, mode='r', shape=(self.dstore_size, 1))
    def __len__(self):
        return self.dstore_size

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.dstore_keys[index])), torch.from_numpy(np.array(self.dstore_vals[index]))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    # method 1: train a lora_linear as an auto-encoder (reconstruction)
    # method 2: train a lora_linear as a word classification
    # method 3: train a lora_linear as a word classification + reconstruction (freezed)
    parser.add_argument("--method", type=int, default=2)
    # path of a training external memory
    parser.add_argument("--train_key_path", default="dstore_keys.npy")
    parser.add_argument("--train_val_path", default="dstore_vals.npy")

    parser.add_argument("--valid_key_path", default="dstore_dev_keys.npy")
    parser.add_argument("--valid_val_path", default="dstore_dev_vals.npy")
    parser.add_argument("--rct_path", default="")

    parser.add_argument("--tensorboard_path", default="")

    parser.add_argument('--train_dstore_size', type=int, default=609011, help='number of items saved in the datastore memmap')
    parser.add_argument('--valid_dstore_size', type=int, default=180851, help='number of items saved in the datastore memmap')

    parser.add_argument('--dstore_dim', type=int, default=2048, help='Size of each key')
    parser.add_argument('--vocab_size', type=int, default=50272)
    parser.add_argument('--save_path', default="./cls_")

    parser.add_argument("--norm_path", default="./opt_last_layer/opt_iml_max_1_3b_layer_norm.pt")
    parser.add_argument("--head_path", default="./opt_last_layer/opt_iml_max_1_3b_lm_head.pt")

    parser.add_argument("--num_rank", default=512, type=int)
    parser.add_argument("--batch_size", default=8000, type=int)

    parser.add_argument("--num_epochs", default=200, type=int)
    parser.add_argument("--kappa", default=0.2, type=float)
    opt =  parser.parse_args()    
    writer = SummaryWriter(opt.tensorboard_path)

    print(opt)

    train_dataset = CustomDataset(opt.train_key_path, opt.train_val_path, opt.train_dstore_size, opt.dstore_dim)
    test_dataset = CustomDataset(opt.valid_key_path, opt.valid_val_path, opt.valid_dstore_size, opt.dstore_dim)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=3, batch_size=opt.batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=3, batch_size=opt.batch_size)

    if opt.method == 1:
        # Method 1: Use lora_linear as an autoencoder
        lora_task_1 = LoraLinear(opt.dstore_dim, opt.dstore_dim, r=opt.num_rank, lora_alpha=1, lora_dropout=0.2).to(device)
        optimizer_1 = torch.optim.Adam(lora_task_1.parameters(), lr=1e-3)
        #train_rct(opt, lora_task_1, optimizer_1, train_batched_dstore_keys, valid_batched_dstore_keys, device)
        train_rct(opt, lora_task_1, optimizer_1, train_dataloader, test_dataloader, device)
    elif opt.method == 2:
        # Method 2: Use lora_linear as a to perform a word classification task
        final_layer_norm = nn.LayerNorm(opt.dstore_dim, elementwise_affine=True).to(device)
        final_layer_norm.load_state_dict(torch.load(opt.norm_path, map_location=device))
        final_layer_norm.requires_grad_(False)
        final_layer_norm = torch.nn.DataParallel(final_layer_norm)
        
        lm_head = nn.Linear(opt.dstore_dim, opt.vocab_size, bias=False).to(device)
        lm_head.load_state_dict(torch.load(opt.head_path, map_location=device))
        lm_head.requires_grad_(False)
        lm_head = torch.nn.DataParallel(lm_head)

        lora_task_2 = LoraLinear(opt.dstore_dim, opt.dstore_dim, r=opt.num_rank, lora_alpha=1, lora_dropout=0.2).to(device)
        optimizer_2 = torch.optim.Adam(lora_task_2.parameters(), lr=1e-3)
        lora_task_2 = torch.nn.DataParallel(lora_task_2)
        train_classification(opt, lora_task_2, lm_head, final_layer_norm, optimizer_2, train_dataloader, test_dataloader, device)

    elif opt.method == 3:
        # Method 3: Use lora_linear and lm_head as a to perform a word classification task + auto-encoder (freezed) model
        lora_task_1 = LoraLinear(opt.dstore_dim, opt.dstore_dim, r=opt.num_rank, lora_alpha=1, lora_dropout=0.2).to(device)
        lora_task_1.load_state_dict(torch.load(opt.rct_path))
        lora_task_1.lora_B.requires_grad_(False)
        lora_task_1 = torch.nn.DataParallel(lora_task_1)

        final_layer_norm = nn.LayerNorm(opt.dstore_dim, elementwise_affine=True).to(device)
        final_layer_norm.load_state_dict(torch.load(opt.norm_path, map_location=device))
        final_layer_norm = torch.nn.DataParallel(final_layer_norm)
        final_layer_norm.requires_grad_(False)

        lm_head = nn.Linear(opt.dstore_dim, opt.vocab_size, bias=False).to(device)
        lm_head.load_state_dict(torch.load(opt.head_path, map_location=device))
        lm_head = torch.nn.DataParallel(lm_head)
        lm_head.requires_grad_(False)

        lora_task_3 = LoraLinear(opt.dstore_dim, opt.dstore_dim, r=opt.num_rank, lora_alpha=1, lora_dropout=0.2).to(device)
        optimizer_3 = torch.optim.Adam(lora_task_3.parameters(), lr=1e-3)
        lora_task_3 = torch.nn.DataParallel(lora_task_3)
        train_rct_classification(opt, lora_task_3, lora_task_1, lm_head, final_layer_norm, optimizer_3, train_dataloader, test_dataloader, device)

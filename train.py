import transformers
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertTokenizer, DistilBertModel
from pylab import rcParams
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from dataloader import create_train_data_loader, create_test_data_loader , get_data_df
from config import get_config
from loss_function import CosineLoss, QuadrupletLoss, TripletLoss, InfoNCELoss
from model import get_model
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# TODO 损失函数改为infoNCE

def train_epoch(model, data_loader, loss_fn, optimizer,scheduler, device, config, textwriter):
    model.train()
    losses = []

    for step, batch in enumerate(data_loader):
        doc_ids = batch["doc_ids"].to(device) # [batch_size,max_length]
        # print("doc_ids:",doc_ids, doc_ids.shape)
        doc_attention_mask = batch["doc_attention_mask"].to(device)
        doc_outputs = model(
            input_ids=doc_ids,
            attention_mask=doc_attention_mask
        ) # [batch_size,embed_dim]
        # print("doc_outputs:",doc_outputs, doc_outputs.shape)

        loss = loss_fn(doc_outputs)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip)
        optimizer.step()
        
        # scheduler.step()

        optimizer.zero_grad()

        if step % config.print_every == 0:
            print(f"[Train] Loss at step {step} = {loss}, lr = {optimizer.state_dict()['param_groups'][0]['lr']}")
            textwriter.write(f"Loss at step {step} = {loss}, lr = {optimizer.state_dict()['param_groups'][0]['lr']} \n")
    return np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples, config,textwriter):
    model.eval()
    correct_predictions = 0
    distances = []
    with torch.no_grad():
        for batch in data_loader:
            doc1_ids = batch["doc1_ids"].to(device) # [batch_size,max_length]
            # print("doc1_ids:",doc1_ids, doc1_ids.shape)
            doc1_attention_mask = batch["doc1_attention_mask"].to(
                device)
            doc2_ids = batch["doc2_ids"].to(device) # [batch_size,max_length]
            # print("doc2_ids:",doc2_ids,doc2_ids.shape)
            doc2_attention_mask = batch["doc2_attention_mask"].to(
                device)
            targets = batch["targets"].to(device) # [batch_size,1]
            # print("targets:",targets,targets.shape)
            doc1_outputs = model(
                input_ids=doc1_ids,
                attention_mask=doc1_attention_mask
            ) # [batch_size,embed_dim]
            print("doc1_outputs:",doc1_outputs,doc1_outputs.shape)
            textwriter.write("doc1_outputs:{} {}\n".format(str(doc1_outputs),str(doc1_outputs.shape)))
            doc2_outputs = model(
                input_ids=doc2_ids,
                attention_mask=doc2_attention_mask
            ) # [batch_size,embed_dim]
            print("doc2_outputs:",doc2_outputs,doc2_outputs.shape)
            textwriter.write("doc2_outputs:{} {}\n".format(str(doc2_outputs),str(doc2_outputs.shape)))
            
            distance_list = []
            for i in range(doc1_outputs.shape[0]):
                distance_list.append(torch.dot(doc1_outputs[i],doc2_outputs[i]))
                distances.append(torch.dot(doc1_outputs[i],doc2_outputs[i]))
            distance = torch.tensor(distance_list,device='cuda')
            print("distance1:",distance,distance.shape)
            textwriter.write("distance1:{} {}\n".format(str(distance),str(distance.shape)))
            
            correct = distance < config.val_threshold
            print("correct:",correct,correct.shape)
            textwriter.write("correct:{} {}\n".format(str(correct),str(correct.shape)))
            
            correct_item = (~(correct ^ targets.bool())).long()
            print("correct_item:",correct_item,correct_item.shape)
            textwriter.write("correct_item:{} {}\n".format(str(correct_item),str(correct_item.shape)))
            correct_sum = torch.sum(correct_item)
            correct_predictions += correct_sum
    return correct_predictions / n_examples, distances

def distance_f(x1,x2):
    distance = (x1-x2).pow(2).sum(1).sqrt()
    distance = 1.0 - (1.0 / (1.0 + distance))
    # distance = torch.dot(x1,x2)
    return distance

if __name__ == '__main__':
    config = get_config()
    # 分布式训练
    # torch.distributed.init_process_group(backend="nccl")
    # local_rank = config.local_rank
    # if local_rank != -1:
    #     print("Uisng Distributed")
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        print('\nGPU is ON!')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Data Loader
    df_train , df_test = get_data_df(config.train_dir, config.val_dir,config)
    df_train = df_train[:10240]
    df_test = df_test[:32]
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    train_data_loader = create_train_data_loader(
        df_train, tokenizer, config.max_len, config.batch_size, mode='train')
    print("训练集:{}个batch".format(len(train_data_loader)))
    test_data_loader = create_test_data_loader(
        df_test, tokenizer, config.max_len, config.batch_size, mode='val')
    print("测试集:{}个batch".format(len(test_data_loader)))

    model = get_model(config)
    model = model.to(device)
    # 分布式训练
    # model = nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],broadcast_buffers=False,find_unused_parameters=True)

    
    if config.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optim == 'amsgrad':
        optimizer = torch.optim.Amsgrad(model.parameters(), lr=config.lr)
    elif config.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr)
    elif config.optim == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.lr)
    total_steps = len(train_data_loader) * config.epochs

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=1000,
    #     num_training_steps=total_steps
    # )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1)
    
    if config.loss_fn == 'triplet':
        # loss_fn = nn.TripletMarginLoss(margin=1, p=2)
        loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_f,margin=1)
    elif config.loss_fn == 'cosine' :
        loss_fn = CosineLoss()
    elif config.loss_fn == 'custom_triplet':
        loss_fn = TripletLoss()
    elif config.loss_fn == 'InfoNCE':
        loss_fn = InfoNCELoss()

    history = {
        'train_acc' : [],
        'train_loss' : [],
        'val_acc' : [],
    }

    best_acc = 0
    config.textfile = open(config.log_dir, "w")

    for epoch in range(config.epochs):
        print(f'Epoch {epoch + 1}/{config.epochs}')
        print('-' * 10)
        config.textfile.write(f"########## Epoch {epoch} ##########\n")

        train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            scheduler,
            device,
            config,
            config.textfile
        )
        # scheduler.step()
        
        print(f'Train loss {train_loss}')
        config.textfile.write(f'Train loss {train_loss}\n')
        val_acc, val_distences = eval_model(
            # model_anchor,
            # model_pos_neg,
            model,
            test_data_loader,
            loss_fn,
            device,
            len(df_test),
            config,
            config.textfile
        )

        print(f'Val accuracy {val_acc}')
        print(f'Val distences {val_distences}')
        config.textfile.write(f'Val accuracy {val_acc}\n')
        config.textfile.write(f'Val distences {val_distences}')

        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)

        # if val_acc > best_acc:
    print('[SAVE] Saving model ... ')
    # torch.save(model_anchor.state_dict(), config.model_path)
    # torch.save(model_pos_neg.state_dict(), config.model_path + '_pos_neg')
    torch.save(model.state_dict(), config.model_path+"_"+str(val_acc))
    best_acc = val_acc
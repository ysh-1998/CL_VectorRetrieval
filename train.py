import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import transformers
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
# torch.set_printoptions(profile="full")


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

from dataloader import get_train_qd_loader,get_test_q_loader,get_test_d_loader
from config import get_config
from loss_function import InfoNCELoss
from model import get_model
from ms_marco_eval import compute_metrics_from_files
import warnings
warnings.filterwarnings("ignore")

# TODO fine tune

def train_epoch(model, qd_loader, loss_fn, optimizer,scheduler, device, config, logfile):
    model.train()
    losses = []

    for step, batch_data in enumerate(qd_loader):
        query_token_ids = batch_data["query_token_ids"].to(device) # [bs,30]
        # print("query_token_ids:", query_token_ids.shape)
        query_attention_mask = batch_data["query_attention_mask"].to(device)
        query_output = model(
            input_ids=query_token_ids,
            attention_mask=query_attention_mask
        ) # [bs,512]
        # print("query_output:",query_output,query_output.shape)
        
        doc_token_ids = batch_data["doc_token_ids"].to(device) # [bs,100]
        # print("doc_token_ids:", doc_token_ids.shape)
        doc_attention_mask = batch_data["doc_attention_mask"].to(device)
        doc_output = model(
            input_ids=doc_token_ids,
            attention_mask=doc_attention_mask
        ) # [bs,512]
        # print("doc_output:",doc_output,doc_output.shape)

        loss = loss_fn(query_output,doc_output)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip)
        optimizer.step()
        
        scheduler.step()

        optimizer.zero_grad()

        if step % config.print_every == 0:
            print(f"[Train] Loss at step {step} = {loss}, lr = {optimizer.state_dict()['param_groups'][0]['lr']}")
            logfile.write(f"Loss at step {step} = {loss}, lr = {optimizer.state_dict()['param_groups'][0]['lr']} \n")
    return np.mean(losses)


def eval_model(model, q_loader, d_loader, device, config, logfile):
    model.eval()
    df_rank = pd.DataFrame(columns=['q_id', 'd_id', 'rank','score'])
    q_id_list = []
    d_id_list = []
    rank = []
    score = []
    with torch.no_grad():
        for i, q_batch in enumerate(q_loader):
            print("querys:第{}/{}个batch".format(i+1,len(q_loader)))
            q_ids = q_batch["ids"]
            query_token_ids = q_batch["token_ids"].to(device) # [bs,30]
            # print("query_token_ids:", query_token_ids.shape)
            query_attention_mask = q_batch["attention_mask"].to(device)
            query_output = model(
                input_ids=query_token_ids,
                attention_mask=query_attention_mask
            ) # [bs,512]
            # print("query_output:",query_output.shape,query_output.device)
            query_output = F.normalize(query_output, dim=1, p=2)
            for j, d_batch in enumerate(d_loader):
                print("querys:第{}/{}个batch".format(i+1,len(q_loader)))
                print("docs:第{}/{}个batch".format(j+1,len(d_loader)))
                d_id = d_batch["ids"].to(device)
                # print("d_id:", len(d_id))
                doc_token_ids = d_batch["token_ids"].to(device) # [bs,100]
                # print("doc_token_ids:", doc_token_ids.shape)
                doc_attention_mask = d_batch["attention_mask"].to(device)
                doc_output = model(
                    input_ids=doc_token_ids,
                    attention_mask=doc_attention_mask
                ) # [bs,512]
                # print("doc_output:",doc_output.shape,doc_output.device)
                doc_output = F.normalize(doc_output, dim=1, p=2)
                rel_score = torch.matmul(query_output, doc_output.transpose(0,1)) # [bs,bs]
                # print("rel_score:",rel_score.shape)
                if j == 0:
                    rel_scores = rel_score
                    d_ids = d_id
                else:
                    rel_scores = torch.cat((rel_scores,rel_score),1) # [qbs,dbs*(j+1)]
                    d_ids = torch.cat((d_ids,d_id)) # [dbs*(j+1)]
            # print("rel_scores:",rel_scores.shape)
            for row in range(rel_scores.shape[0]):
                unrank_doc_rel = rel_scores[row].cpu().numpy()
                unrank_id = d_ids.cpu().numpy()
                q_id = q_ids[row].cpu().numpy()
                df_docs = pd.DataFrame({0:unrank_id,1:unrank_doc_rel})
                df_docs = df_docs.sort_values(by=1,ascending=False)
                rank_doc_id = df_docs[0].values.tolist()[:100]
                rank_score = df_docs[1].values.tolist()[:100]
                q_id_list += [q_id]*100
                d_id_list += rank_doc_id
                rank += range(1,101)
                score += rank_score
        df_rank['q_id'] = q_id_list
        df_rank['d_id'] = d_id_list
        df_rank['rank'] = rank
        df_rank['score'] = score
    return df_rank

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
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    # Train Data Loader
    df_train_qds = pd.read_csv(config.train_qd_dir, sep='\t',header=None)
    df_train_qds = df_train_qds[:40960]
    train_qd_loader = get_train_qd_loader(
        df_train_qds,tokenizer,config.q_max_len,config.d_max_len,config.batch_size,mode='train')
    print(f"train_qd_pairs: {len(df_train_qds)},train_batchs:{len(train_qd_loader)}")
    
    # Test Data Loader
    df_test_queries = pd.read_csv(config.test_querys_dir, sep='\t',header=None)
    df_test_docs = pd.read_csv(config.test_docs_dir, sep='\t',header=None)
    test_q_loader = get_test_q_loader(df_test_queries,tokenizer,config.q_max_len,config.q_batch_size,mode='val')
    test_d_loader = get_test_d_loader(df_test_docs,tokenizer,config.d_max_len,config.d_batch_size,mode='val')
    print(f"test_q: {len(df_test_queries)},test_q_batchs:{len(test_q_loader)}")
    print(f"test_d: {len(df_test_docs)},test_d_batchs:{len(test_d_loader)}")

    model = get_model(config)
    model = model.to(device)
    print("加载模型")
    # model.load_state_dict(torch.load(config.model_path))
    model.load_state_dict(torch.load("ckpt/0.9201_model_state.bin"))
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
    total_steps = len(train_qd_loader) * config.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps//10,
        num_training_steps=total_steps
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1)
    
    if config.loss_fn == 'triplet':
        # loss_fn = nn.TripletMarginLoss(margin=1, p=2)
        loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=distance_f,margin=1)
    elif config.loss_fn == 'InfoNCE':
        loss_fn = InfoNCELoss()

    history = {
        'train_acc' : [],
        'train_loss' : [],
        'val_acc' : [],
    }

    logfile = open(config.log_dir, "w")
    best_MRR = 0
    for epoch in range(config.epochs):
        print(f'Epoch {epoch + 1}/{config.epochs}')
        print('-' * 10)
        logfile.write(f"########## Epoch {epoch} ##########\n")

        train_loss = train_epoch(
            model,
            train_qd_loader,
            loss_fn,
            optimizer,
            scheduler,
            device,
            config,
            logfile
        )
        # scheduler.step()
        print(f'Train loss {train_loss}')
        logfile.write(f'Train loss {train_loss}\n')
        
        qd_rank = eval_model(
            model,
            test_q_loader,
            test_d_loader,
            device,
            config,
            logfile
        )
        qd_rank.to_csv("result/qd_rank.tsv",sep='\t',index=False,header=False)
        metrics = compute_metrics_from_files(config.test_qrels_dir, "result/qd_rank.tsv")
        print('#####################')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))
            logfile.write('{}: {}\n'.format(metric, metrics[metric]))
        print('#####################')
        MRR = round(metrics['MRR @10'],4)
        if MRR > best_MRR:
            best_MRR = MRR
            print('[SAVE] Saving model ... ')
            torch.save(model.state_dict(), config.model_path+"_"+str(MRR))
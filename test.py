import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import transformers
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from model import get_model
from config import get_config
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from dataloader import get_data_df,get_query_data_loader,get_doc_data_loader
import torch.nn.functional as F

def to_vector(model, batch_data, device,config):
    model.eval()
    with torch.no_grad():
        token_ids = batch_data["token_ids"].to(device) # [bs,30]or[bs,100]
        print("token_ids:", token_ids.shape)
        attention_mask = batch_data["attention_mask"].to(
            device)
        vector = model(
            input_ids=token_ids,
            attention_mask=attention_mask
        ) # [bs,512]
        vector = F.normalize(vector, dim=1, p=2)
        print("vector:",vector.shape,vector.device)

    return vector

if __name__ == '__main__':
    config = get_config()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        print('\nGPU is ON!')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    df_queries,df_docs,df_qrels = get_data_df(config.querys_dir,config.docs_dir,config.qrels_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    q_data_loader = get_query_data_loader(df_queries,tokenizer,config.max_len,config.q_batch_size)
    d_data_loader = get_doc_data_loader(df_docs,tokenizer,config.max_len,config.batch_size)
    
    model = get_model(config)
    model = model.to(device)
    print("加载模型")
    model.load_state_dict(torch.load("ckpt/0.9435_model_state.bin"))
    df_rank = pd.DataFrame(columns=['q_id', 'd_id', 'rank'])
    q_id_list = []
    d_id_list = []
    rank = []
    for i, q_batch in enumerate(q_data_loader):
        q_ids = q_batch["ids"].cpu().numpy().tolist()
        q_vector = to_vector(model,q_batch,device,config)
        print("q_vector:", q_vector.shape,q_vector.device) # [qbs,512]
        for j, d_batch in enumerate(d_data_loader):
            d_vector = to_vector(model,d_batch,device,config)
            print("d_vector:", d_vector.shape,d_vector.device) # [dbs,512]
            rel_score = torch.matmul(q_vector, d_vector.transpose(0,1)) # [qbs,dbs]
            if j == 0:
                rel_scores = rel_score
            else:
                rel_scores = torch.cat((rel_scores,rel_score),1) # [qbs,dbs*(j+1)]
        for row in rel_scores.shape[0]:
            unrank_doc = rel_scores[row].cpu().numpy()
            df_docs[2] = unrank_doc
            df_docs = df_docs.sort_values(by=2,ascending=False)
            rand_doc = df_docs[1].values.tolist()[:100]
            q_id_list += [q_ids[row]]*100
            d_id_list += rand_doc
            rank += range(1,101)
    df_rank['q_id'] = q_id_list
    df_rank['d_id'] = d_id_list
    df_rank['rank'] = rank           
    
    df_rank.to_csv("qd_rank.tsv",sep='\t',index=False,header=False)
    
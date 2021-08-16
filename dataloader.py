import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from gensim.parsing.preprocessing import remove_stopwords

class TrainQDDataset(Dataset):
    def __init__(self, q_ids, queries, d_ids, docs, tokenizer, q_max_len,d_max_len):
        self.q_ids = q_ids
        self.queries = queries
        self.d_ids = d_ids
        self.docs = docs
        self.tokenizer = tokenizer
        self.q_max_len = q_max_len
        self.d_max_len = d_max_len
    
    def __len__(self):
        return len(self.q_ids)

    def __getitem__(self, item):
        q_id = self.q_ids[item]
        d_id = self.d_ids[item]
        query = self.queries[item]
        doc = self.docs[item]
        
        q_encoding = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=self.q_max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        d_encoding = self.tokenizer(
            doc,
            add_special_tokens=True,
            max_length=self.d_max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            "q_id":q_id,
            'query': query,
            "d_id":d_id,
            'doc': doc,
            'query_token_ids': q_encoding['input_ids'].flatten(),
            'query_attention_mask': q_encoding['attention_mask'].flatten(),
            'doc_token_ids': d_encoding['input_ids'].flatten(),
            'doc_attention_mask': d_encoding['attention_mask'].flatten(),
        }

def get_train_qd_loader(df_qds,tokenizer,q_max_len,d_max_len,batch_size,mode='train'):
    ds = TrainQDDataset(
        q_ids=df_qds[0].values.tolist(),
        queries=df_qds[1].values.tolist(),
        d_ids=df_qds[2].values.tolist(),
        docs=df_qds[3].values.tolist(),
        tokenizer=tokenizer,
        q_max_len=q_max_len,
        d_max_len=d_max_len
    )
    if mode == 'train':
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        )

class TestQDDataset(Dataset):
    def __init__(self, ids, texts, tokenizer, max_len):
        self.ids = ids
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        text = self.texts[item]
        id = self.ids[item]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        return {
            "ids":id,
            'text': text,
            'token_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

def get_test_q_loader(df_queries,tokenizer,max_len,batch_size,mode='train'):
    ds = TestQDDataset(
        ids=df_queries[0].values.tolist(),
        texts=df_queries[1].values.tolist(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
    )

def get_test_d_loader(df_docs,tokenizer,max_len,batch_size,mode='train'):
    ds = TestQDDataset(
        ids=df_docs[0].values.tolist(),
        texts=df_docs[1].values.tolist(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
    )
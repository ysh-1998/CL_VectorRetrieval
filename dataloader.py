import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class TestDataset(Dataset):

    def __init__(self, doc_1, doc_2, targets, tokenizer, max_len):
        self.doc_1 = doc_1
        self.doc_2 = doc_2
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.doc_1)

    def __getitem__(self, item):

        doc_1 = str(self.doc_1[item])
        doc_2 = str(self.doc_2[item])
        target = self.targets[item]

        doc1_encoding = self.tokenizer.encode_plus(
            doc_1,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        doc2_encoding = self.tokenizer.encode_plus(
            doc_2,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        return {
            'doc1': doc_1,
            'doc2': doc_2,
            'doc1_ids': doc1_encoding['input_ids'].flatten(),
            'doc1_attention_mask': doc1_encoding['attention_mask'].flatten(),
            'doc2_ids': doc2_encoding['input_ids'].flatten(),
            'doc2_attention_mask': doc2_encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class TrainDataset(Dataset):

    def __init__(self, doc, tokenizer, max_len):
        self.doc = doc
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.doc)

    def __getitem__(self, item):

        doc = str(self.doc[item])

        doc_encoding = self.tokenizer.encode_plus(
            doc,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        return {
            'doc': doc,
            'doc_ids': doc_encoding['input_ids'].flatten(),
            'doc_attention_mask': doc_encoding['attention_mask'].flatten(),
        }


def create_test_data_loader(df, tokenizer, max_len, batch_size, mode='train'):
    ds = TestDataset(
        doc_1=df.doc_1.to_numpy(),
        doc_2=df.doc_2.to_numpy(),
        targets=df.is_duplicate.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
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


def create_train_data_loader(df, tokenizer, max_len, batch_size, mode='train'):
    ds = TrainDataset(
        doc=df.doc.to_numpy(),
        # positive=df.positive.to_numpy(),
        # negative=df.negative.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
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

def get_data_df(train_dir,test_dir,config):
    df_test = pd.read_csv(test_dir, sep='\t')
    # df_test2 = pd.read_csv('../test_final.csv')
    # df_test = df_test[['question1','question2','is_duplicate']]
    # df_test = pd.concat([df_test,df_test2])


    # Train data
    # df_train = pd.read_csv('../test_final.csv') #
    # df_train2 = pd.read_csv('../3254_train.csv') #
    # df_train3 = pd.read_csv('../600_train.csv') #
    # df_train3 = df_train3.drop('Unnamed: 0',axis=1)
    # df_train4 = pd.read_csv('../9303_train.csv') #
    # df_train4.sample(frac=1)
    # df_train4 = df_train4[:3000]
    # df_train = pd.concat([df_train,df_train2,df_train3,df_train4])

    #Train data triplet

    df_train = pd.read_csv(train_dir, sep='\t')
    
    # if config.use_aug_data == True:
    #   df_train2 = pd.read_csv('./data/triplet_data.csv')
    #   df_train2 = df_train2.drop('Unnamed: 0',axis=1)
    #   df_train = pd.concat([df_train,df_train2])

    print("数据集尺寸: ", df_train.shape, df_test.shape)
    return df_train, df_test
import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup



class SimilarityClassifier(nn.Module):
    def __init__(self, PRE_TRAINED_MODEL_NAME, embed_dim, dropout_p, freeze=False):
        super(SimilarityClassifier, self).__init__()
        # self.bert = DistilBertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]
        # bert_output = torch.mean(bert_output, dim=1)
        # out = self.drop(bert_output)
        bert_output = bert_output[:,0]
        out = bert_output

        return out

def get_model(config):
    model = SimilarityClassifier(config.PRE_TRAINED_MODEL_NAME, config.embed_dim, config.dropout,config.freeze)
    return model
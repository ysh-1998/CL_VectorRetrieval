import torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    def __init__(self,lamda=0.05):
        super(InfoNCELoss, self).__init__()
        self.lamda = lamda
    def forward(self,q_emb,d_emb):
        batch_size = q_emb.size(0)
        q_emb = F.normalize(q_emb, dim=1, p=2)
        d_emb = F.normalize(d_emb, dim=1, p=2)
        batch_label = torch.arange(0, batch_size, dtype=torch.long,device='cuda')
        # batch_label = torch.cat([torch.arange(1, batch_size, step=2, dtype=torch.long,device='cuda').unsqueeze(1),
        #                 torch.arange(0, batch_size, step=2, dtype=torch.long,device='cuda').unsqueeze(1)],
        #                dim=1).reshape([batch_size,])
        sim_score = torch.matmul(q_emb, d_emb.transpose(0,1))
        # print("sim_score:",sim_score, sim_score.shape)
        # print("batch_label:",batch_label, batch_label.shape)
        # sim_score = sim_score - torch.eye(batch_size,device='cuda') * 1e12
        sim_score = sim_score / self.lamda
        losses = F.cross_entropy(sim_score,batch_label)
        return losses.mean()



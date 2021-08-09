import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self,lamda=0.05):
        super(InfoNCELoss, self).__init__()
        self.lamda = lamda
    def forward(self,batch_emb):
        batch_size = batch_emb.size(0)
        batch_emb = F.normalize(batch_emb, dim=1, p=2)
        batch_label = torch.cat([torch.arange(1, batch_size, step=2, dtype=torch.long,device='cuda').unsqueeze(1),
                        torch.arange(0, batch_size, step=2, dtype=torch.long,device='cuda').unsqueeze(1)],
                       dim=1).reshape([batch_size,])
        sim_score = torch.matmul(batch_emb, batch_emb.transpose(0,1))
        sim_score = sim_score - torch.eye(batch_size,device='cuda') * 1e12
        sim_score = sim_score / self.lamda
        print("sim_score:",sim_score, sim_score.shape)
        losses = F.cross_entropy(sim_score,batch_label)
        return losses.mean()



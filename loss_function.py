import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuadrupletLoss:
    def __init__(self, margin_a=1, margin_b=0.5, p=2):
        self.margin_a = margin_a
        self.margin_b = margin_b
        self.p = p

    def __call__(self, output_a, output_p, output_n1, output_n2):
        distance_1 = (output_a - output_p) ** 2 - \
            (output_a - output_n1) ** 2 + self.margin_a
        distance_1 = max(distance_1, 0)

        distance_2 = (output_a - output_p) ** 2 - \
            (output_n1 - output_n2) ** 2 + self.margin_b
        distance_2 = max(distance_2, 0)

        total_distance = distance_1 + distance_2

        return total_distance


class CosineLoss:
    def __init__(self, margin_a=1, margin_b=0.5, p=2):
        self.margin_a = margin_a
        self.margin_b = margin_b
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)

    def __call__(self, output_a, output_p, output_n1):
        distance_1 = self.cosine(output_a, output_p) - \
            self.cosine(output_a, output_n1) + self.margin_a
        distance_1 = max(torch.mean(distance_1), 0)

        total_distance = distance_1

        return total_distance

class TripletLoss(nn.Module):
    def __init__(self, margin = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)

    def calc_euclidean(self, x1, x2):
        return(x1 - x2).pow(2).sum(1)

    def calc_cosine(self,x1,x2):
        return self.cosine(x1,x2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative_a = self.calc_euclidean(anchor, negative)
        distance_negative_b = self.calc_euclidean(positive, negative)
        losses = torch.relu(distance_positive - (distance_negative_a + distance_negative_b)/2.0 + self.margin)
        return losses.mean()

class InfoNCELoss(nn.Module):
    def __init__(self,lamda=0.05):
        super(InfoNCELoss, self).__init__()
        self.lamda = lamda
    def forward(self,batch_emb):
        batch_size = batch_emb.size(0)
        batch_label = torch.cat([torch.arange(1, batch_size, step=2, dtype=torch.long,device='cuda').unsqueeze(1),
                        torch.arange(0, batch_size, step=2, dtype=torch.long,device='cuda').unsqueeze(1)],
                       dim=1).reshape([batch_size,])
        sim_score = torch.matmul(batch_emb, batch_emb.transpose(0,1))
        sim_score = sim_score - torch.eye(batch_size,device='cuda') * 1e12
        sim_score = sim_score / self.lamda
        losses = F.cross_entropy(sim_score,batch_label)
        return losses.mean()



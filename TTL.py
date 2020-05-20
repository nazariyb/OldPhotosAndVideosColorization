import torch
import numpy as np

from utils import KL_divergence
from device2use import device


def _loss(memory, resnet_out, color_feature, loss_threshold):
    bs = resnet_out.size()[0]
    cosine_score = torch.matmul(resnet_out, torch.t(memory.K))

    top_k_score, top_k_index = torch.topk(cosine_score, memory.top_k, 1)

    ### For unsupervised training
    color_value_expand = torch.unsqueeze(torch.t(memory.V), 0)
    color_value_expand = torch.cat([color_value_expand[:,:,idx] for idx in top_k_index], dim = 0)

    color_feat_expand = torch.unsqueeze(color_feature, 2)
    color_feat_expand = torch.cat([color_feat_expand for _ in range(memory.top_k)], dim = 2)

    color_similarity = KL_divergence(color_value_expand, color_feat_expand, 1)

    loss_mask = color_similarity < loss_threshold
    loss_mask = torch.from_numpy(loss_mask).float().to(device)

    pos_score, pos_index = torch.topk(torch.mul(top_k_score, loss_mask), 1, dim = 1)
    neg_score, neg_index = torch.topk(torch.mul(top_k_score, 1 - loss_mask), 1, dim = 1)

    hinge = torch.clamp(neg_score - pos_score + .1, min = 0.0)
    loss = torch.mean(hinge)

    return loss


def TTL(memory):
    return lambda resnet_out, color_feature, loss_threshold: \
                    _loss(memory, resnet_out, color_feature, loss_threshold)
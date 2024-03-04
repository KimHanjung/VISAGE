import logging
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.layers import Conv2d, get_norm

from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MLP
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

class AppearanceDecoder(nn.Module):
    def __init__(
        self, 
        hidden_dim,
        in_channels,
        reid_weight,
        aux_reid_weight,
        app_reid_weight,
        app_aux_reid_weight,
    ):
        super().__init__()
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # only use res2 feature
        self.num_feature_levels = len(in_channels)
        self.input_proj = nn.ModuleList()
        for l in range(self.num_feature_levels):
            self.input_proj.append(nn.Sequential(
                Conv2d(in_channels[l], hidden_dim, kernel_size=1, norm=get_norm("GN", hidden_dim)),
            ))
        
        self.appearance_embd = MLP(hidden_dim * len(in_channels), hidden_dim, hidden_dim, 3)
        self.reid_embd = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        self.reid_weight = reid_weight
        self.aux_reid_weight = aux_reid_weight
        self.app_reid_weight = app_reid_weight
        self.app_aux_reid_weight = app_aux_reid_weight

    def forward(self, pred_embds, appearance_features, output_masks, indices=None, targets=None):
        B, T, Q, C = pred_embds.shape
        output_masks = output_masks.squeeze(2).detach()

        appearance_queries = []
        for i in range(self.num_feature_levels):
            resize_mask = (F.interpolate(output_masks, size=appearance_features[i].shape[-2:], mode='bilinear', align_corners=False).sigmoid() > 0.5)
            proj_features = self.input_proj[i](appearance_features[i])
            mask_pool_features = torch.einsum('bqd,bcd->bqc', resize_mask.float().flatten(2), proj_features.flatten(2)) / (resize_mask.sum(dim=(2, 3))[..., None] + 1e-6)
            appearance_queries.append(mask_pool_features)
        appearance_queries = self.appearance_embd(torch.cat(appearance_queries, dim=-1)).reshape(B, T, Q, -1)

        reid_queries = self.reid_embd(pred_embds.flatten(0, 1)).reshape(B, T, Q, -1)

        if self.training:
            sample_appearance_queries = appearance_queries.unbind(1)
            sample_reid_queries = reid_queries.unbind(1)
            valid_indices = [t['ids'].squeeze(1) != -1 for t in targets]

            losses = dict()
            total_loss_reid = 0
            total_loss_aux_cos = 0
            total_loss_reid_app = 0
            total_loss_aux_cos_app = 0
            num_instances = 0

            for t in range(1, T):
                sample_valid_indices = [(v1 & v2).cpu() for v1, v2 in zip(valid_indices[t-1::T], valid_indices[t::T])]
                key_indices = [src[tgt.argsort()] for src, tgt in indices[t-1::T]]
                ref_indices = [src[tgt.argsort()] for src, tgt in indices[t::T]]

                # object query contrastive loss
                key_reid_queries = sample_reid_queries[t-1]
                ref_reid_queries = sample_reid_queries[t]
                
                dists, cos_dists, labels = self.match(key_reid_queries, ref_reid_queries, ref_indices, key_indices, sample_valid_indices)
                loss_reid, loss_aux_cos = self.loss(dists, cos_dists, labels)
                total_loss_reid += loss_reid
                total_loss_aux_cos += loss_aux_cos

                # appearance query contrastive loss
                key_appearance_queries = sample_appearance_queries[t-1]
                ref_appearance_queries = sample_appearance_queries[t]

                dists, cos_dists, labels = self.match(key_appearance_queries, ref_appearance_queries, ref_indices, key_indices, sample_valid_indices)
                loss_reid_app, loss_aux_cos_app = self.loss(dists, cos_dists, labels)
                total_loss_reid_app += loss_reid_app
                total_loss_aux_cos_app += loss_aux_cos_app

                num_instances += sum([len(dist) for dist in dists])

            if num_instances == 0:
                losses = {f'loss_reid': reid_queries.sum()*0., f'loss_aux_reid': reid_queries.sum()*0.,
                          f'loss_reid_app': appearance_queries.sum()*0., f'loss_aux_reid_app': appearance_queries.sum()*0.}
            else:
                losses = {f'loss_reid': total_loss_reid / num_instances * self.reid_weight, 
                          f'loss_aux_reid': total_loss_aux_cos / num_instances * self.aux_reid_weight,
                          f'loss_reid_app': total_loss_reid_app / num_instances * self.app_reid_weight, 
                          f'loss_aux_reid_app': total_loss_aux_cos_app / num_instances * self.app_aux_reid_weight}
            return losses

        return reid_queries, appearance_queries
    
    def match(self, key, ref, ref_indices, key_indices, valid_indices, appearance=False):
        dists, cos_dists, labels = [], [], []
        if appearance:
            split_idx = [len(src) for src in valid_indices]
            key = key.split(split_idx)
            ref = ref.split(split_idx)

        for key_embed, ref_embed, key_idx, ref_idx, valid_idx in zip(key, ref, key_indices, ref_indices, valid_indices):
            if not appearance:
                anchor = key_embed[key_idx[valid_idx]] 
                target = ref_embed[ref_idx[valid_idx]]
            else:
                anchor = key_embed
                target = ref_embed
            dist = torch.einsum('ac, kc -> ak', anchor, target)
            cos_dist = torch.einsum('ac, kc -> ak', F.normalize(anchor, dim=-1), F.normalize(target, dim=-1))
            label_ = torch.eye(len(dist), device=dist.device)

            dists.append(dist)
            cos_dists.append(cos_dist)
            labels.append(label_)

        return dists, cos_dists, labels
    
    def loss(self, dists, cos_dists, labels):
        
        loss_reid = 0.0
        loss_aux_cos = 0.0

        for dist, cos_dist, label in zip(dists, cos_dists, labels):
            pos_inds = (label == 1)
            neg_inds = (label == 0)
            dist_pos = dist * pos_inds.float()
            dist_neg = dist * neg_inds.float()
            dist_pos[neg_inds] = dist_pos[neg_inds] + float('inf')
            dist_neg[pos_inds] = dist_neg[pos_inds] - float('inf')

            _pos_expand = torch.repeat_interleave(dist_pos, dist.shape[1], dim=1)
            _neg_expand = dist_neg.repeat(1, dist.shape[1])

            x = F.pad((_neg_expand - _pos_expand), (0,1), value=0)
            loss = torch.logsumexp(x, dim=1)# * (dist.shape[0] > 0).float()
            loss_reid += loss.sum()

            loss = torch.abs(cos_dist - label.float())**2
            loss_aux_cos += loss.mean(-1).sum()

        return loss_reid, loss_aux_cos
    
# Modified by Hanjung Kim from: https://github.com/NVlabs/MinVIS

# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import math
from typing import Tuple
import einops

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former_video.modeling.criterion import VideoSetCriterion
from mask2former_video.modeling.matcher import VideoHungarianMatcher
from mask2former_video.utils.memory import retry_if_cuda_oom

from .appearance_decoder import AppearanceDecoder

from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

class Memorybank(object):
    def __init__(self, num_queries, hidden_dim, bank_size=3):
        self.bank_size = bank_size
        self.num_queries = num_queries
        self.size = 0

        self.memory_bank = torch.zeros(self.bank_size, num_queries, hidden_dim).cuda()
        self.scores = torch.zeros(self.bank_size, num_queries).cuda()

    def reset(self):
        self.memory_bank.zero_()
        self.scores.zero_()
        self.size = 0

    def update(self, embeddings, max_scores):
        '''
        Args:
            embeddings: (Q, C)
        '''
        self.memory_bank = self.memory_bank.roll(1, dims=0)
        self.memory_bank[0] = embeddings
        self.scores = self.scores.roll(1, dims=0)
        self.scores[0] = max_scores
        self.size = min(self.size + 1, self.bank_size)

    def get(self):
        temporal_weight = torch.arange(0, 1+1e-6, 1 / self.size, device=self.memory_bank.device)[1:]
        score_weight = self.scores[:self.size]
        weight = temporal_weight[:, None] + score_weight
        temporal_embedding = (self.memory_bank[:self.size] * weight[..., None]).sum(0) / weight.sum(0)[:, None]

        return temporal_embedding

@META_ARCH_REGISTRY.register()
class VISAGE(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        appearance_decoder: nn.Module,
        num_queries: int,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        appearance_weight: float,
        # video
        num_frames,
        num_classes,
        hidden_dim,
        appearance_in_features,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames

        self.num_classes = num_classes

        self.appearance_weight = appearance_weight
        self.appearance_decoder = appearance_decoder
        self.appearance_in_features = appearance_in_features

        self.memory_bank = Memorybank(num_queries, hidden_dim=hidden_dim, bank_size=5)
        self.appearance_memory_bank = Memorybank(num_queries, hidden_dim=hidden_dim, bank_size=5)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        
        appearance_in_features = cfg.MODEL.APPEARANCE_DECODER.IN_FEATURES
        appearance_in_channels = [backbone.output_shape()[f].channels for f in appearance_in_features]

        appearance_decoder = AppearanceDecoder(
            hidden_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            in_channels=appearance_in_channels,
            reid_weight=cfg.MODEL.APPEARANCE_DECODER.REID_WEIGHT,
            aux_reid_weight=cfg.MODEL.APPEARANCE_DECODER.AUX_REID_WEIGHT,
            app_reid_weight=cfg.MODEL.APPEARANCE_DECODER.APP_REID_WEIGHT,
            app_aux_reid_weight=cfg.MODEL.APPEARANCE_DECODER.APP_AUX_REID_WEIGHT,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "appearance_decoder": appearance_decoder,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "appearance_weight": cfg.MODEL.APPEARANCE_WEIGHT,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "hidden_dim": cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            "appearance_in_features": appearance_in_features,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if self.training:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)

            outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)

            # bipartite matching-based loss
            losses, indices = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            appearance_features = [features[f].detach() for f in self.appearance_in_features]
            appearance_loss = self.appearance_decoder(outputs['pred_embds'], appearance_features, outputs['pred_masks'], indices, targets)
            losses.update(appearance_loss)
                    
            return losses
        else:
            outputs, reid_queries, appearance_queries = self.run_window_inference(images.tensor)
            outputs = self.post_processing(outputs, reid_queries, appearance_queries)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size)

    def frame_decoder_loss_reshape(self, outputs, targets):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> (b t) q () h w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_logits'], 'b t q c -> (b t) q c'
                )

        gt_instances = []
        for targets_per_video in targets:
            # labels: N (num instances)
            # ids: N, num_labeled_frames
            # masks: N, num_labeled_frames, H, W
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]

                invalid_idx = masks.sum(dim=(1,2,3)) == 0
                labels = targets_per_video['labels']
                labels[invalid_idx] = self.num_classes
                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})

        return outputs, gt_instances

    def match_from_embds(self, prevs, curs):
        cur_embds, cur_app_embds = curs
        tgt_embds, tgt_app_embds = prevs

        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cur_app_embds = cur_app_embds / cur_app_embds.norm(dim=1)[:, None]
        tgt_app_embds = tgt_app_embds / tgt_app_embds.norm(dim=1)[:, None]
        cos_sim_app = torch.mm(cur_app_embds, tgt_app_embds.transpose(0,1))

        cost_embd = (1 - self.appearance_weight) * cos_sim + self.appearance_weight * cos_sim_app

        C = 1.0 * (cost_embd)
        C = C.cpu()
        
        indices = linear_sum_assignment(C.transpose(0, 1), maximize=True)  # target x current
        indices = indices[1]  # permutation that makes current aligns to target

        return indices

    def post_processing(self, outputs, reid_queries, appearance_queries):
        pred_logits, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embds']
        self.memory_bank.reset()
        self.appearance_memory_bank.reset()

        # pred_logits: 1 t q c
        # pred_masks: 1 q t h w
        pred_logits = pred_logits[0]
        pred_masks = einops.rearrange(pred_masks[0], 'q t h w -> t q h w')
        pred_embds = pred_embds[0]
        reid_queries = reid_queries[0]
        appearance_queries = appearance_queries[0]

        pred_logits = list(torch.unbind(pred_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embds = list(torch.unbind(pred_embds))
        reid_queries = list(torch.unbind(reid_queries))
        appearance_queries = list(torch.unbind(appearance_queries))

        out_logits = []
        out_masks = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        max_scores, _ = torch.max(out_logits[-1].softmax(dim=-1)[:, :-1], dim=-1)
        self.memory_bank.update(reid_queries[0], max_scores)
        self.appearance_memory_bank.update(appearance_queries[0], max_scores)

        for i in range(1, len(pred_logits)):
            prevs = (self.memory_bank.get(), self.appearance_memory_bank.get())
            curs = (reid_queries[i], appearance_queries[i])
            indices = self.match_from_embds(prevs, curs)

            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            max_scores, _ = torch.max(out_logits[-1].softmax(dim=-1)[:, :-1], dim=-1)
            self.memory_bank.update(reid_queries[i][indices, :], max_scores)
            self.appearance_memory_bank.update(appearance_queries[i][indices, :], max_scores)

        out_logits = sum(out_logits)/len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)

        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks

        return outputs

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        reid_list = []
        appearance_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            appearance_features = [features[f].detach() for f in self.appearance_in_features]
            reid_queries, appearance_queries = self.appearance_decoder(out['pred_embds'], appearance_features, einops.rearrange(out['pred_masks'], 'b q t h w -> (b t) q () h w'))
            del features['res2'], features['res3'], features['res4'], features['res5'], appearance_features
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            out_list.append(out)
            reid_list.append(reid_queries)
            appearance_list.append(appearance_queries)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2).detach()
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=1).detach()
        reid_queries = torch.cat(reid_list, dim=1).detach()
        appearance_queries = torch.cat(appearance_list, dim=1).detach()

        return outputs, reid_queries, appearance_queries

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            gt_classes_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_classes_per_video.append(targets_per_frame.gt_classes)
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                else:
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = torch.stack(gt_classes_per_video).min(dim=0)[0]
            gt_classes_per_video = gt_classes_per_video[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                          # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(len(pred_cls), 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.sem_seg_head.num_classes, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]

            interim_mask_soft = pred_masks.sigmoid()
            interim_mask_hard = interim_mask_soft > 0.5

            numerator   = (interim_mask_soft.flatten(1) * interim_mask_hard.flatten(1)).sum(1)
            denominator = interim_mask_hard.flatten(1).sum(1)

            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            scores_per_image *= (numerator / (denominator + 1e-6))
            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output

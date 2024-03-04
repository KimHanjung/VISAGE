import copy
import logging
import random
import numpy as np
from typing import List, Union
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
)

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.data import MapDataset

from .augmentation import build_augmentation, build_pseudo_augmentation
from .datasets.ytvis import COCO_TO_YTVIS_2019, COCO_TO_YTVIS_2021, COCO_TO_OVIS
from pycocotools import mask as coco_mask

__all__ = ["YTVISDatasetMapper", "CocoClipDatasetMapper"]


def filter_empty_instances(instances, by_box=True, by_mask=True, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())
    r.append(instances.gt_classes != -1)

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x

    instances.gt_ids[~m] = -1
    return instances


def _get_dummy_anno(num_classes):
    return {
        "iscrowd": 0,
        "category_id": num_classes,
        "id": -1,
        "bbox": np.array([0, 0, 0, 0]),
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [np.array([0.0] * 6)]
    }


def ytvis_annotations_to_instances(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_ids",
            "gt_masks", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    ids = [int(obj["id"]) for obj in annos]
    ids = torch.tensor(ids, dtype=torch.int64)
    target.gt_ids = ids

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = BitMasks(
            torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        )
        target.gt_masks = masks

    return target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class YTVISDatasetMapper:
    """
    A callable which takes a dataset dict in YouTube-VIS Dataset format,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        is_tgt: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        sampling_frame_shuffle: bool = False,
        num_classes: int = 40,
        src_dataset_name: str = "",
        tgt_dataset_name: str = "",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
        """
        # fmt: off
        self.is_train               = is_train
        self.is_tgt                 = is_tgt
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.sampling_frame_shuffle = sampling_frame_shuffle
        self.num_classes            = num_classes

        if not is_tgt:
            self.src_metadata = MetadataCatalog.get(src_dataset_name)
            self.tgt_metadata = MetadataCatalog.get(tgt_dataset_name)
            if tgt_dataset_name.startswith("ytvis_2019"):
                src2tgt = OVIS_TO_YTVIS_2019
            elif tgt_dataset_name.startswith("ytvis_2021"):
                src2tgt = OVIS_TO_YTVIS_2021
            elif tgt_dataset_name.startswith("ovis"):
                if src_dataset_name.startswith("ytvis_2019"):
                    src2tgt = YTVIS_2019_TO_OVIS
                elif src_dataset_name.startswith("ytvis_2021"):
                    src2tgt = YTVIS_2021_TO_OVIS
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            self.src2tgt = {}
            for k, v in src2tgt.items():
                self.src2tgt[
                    self.src_metadata.thing_dataset_id_to_contiguous_id[k]
                ] = self.tgt_metadata.thing_dataset_id_to_contiguous_id[v]

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, is_tgt: bool = True):
        augs = build_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        sampling_frame_shuffle = cfg.INPUT.SAMPLING_FRAME_SHUFFLE

        ret = {
            "is_train": is_train,
            "is_tgt": is_tgt,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "sampling_frame_shuffle": sampling_frame_shuffle,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "tgt_dataset_name": cfg.DATASETS.TRAIN[-1],
        }

        return ret

    def __call__(self, dataset_dict, paste_dataset_dict=None):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in YTVIS Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # TODO consider examining below deepcopy as it costs huge amount of computations.
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict["length"]
        if self.is_train:
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
            if self.sampling_frame_shuffle:
                random.shuffle(selected_idx)
        else:
            selected_idx = range(video_length)

        video_annos = dataset_dict.pop("annotations", None)
        file_names = dataset_dict.pop("file_names", None)

        if self.is_train:
            _ids = set()
            for frame_idx in selected_idx:
                _ids.update([anno["id"] for anno in video_annos[frame_idx]])
            ids = dict()
            for i, _id in enumerate(_ids):
                ids[_id] = i

        dataset_dict["video_len"] = len(video_annos)
        dataset_dict["frame_idx"] = list(selected_idx)
        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = []
        for frame_idx in selected_idx:
            dataset_dict["file_names"].append(file_names[frame_idx])

            # Read image
            image = utils.read_image(file_names[frame_idx], format=self.image_format)
            utils.check_image_size(dataset_dict, image)

            aug_input = T.AugInput(image)
            transforms = self.augmentations(aug_input)
            image = aug_input.image

            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"].append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))))

            if (video_annos is None) or (not self.is_train):
                continue

            # NOTE copy() is to prevent annotations getting changed from applying augmentations
            _frame_annos = []
            for anno in video_annos[frame_idx]:
                _anno = {}
                for k, v in anno.items():
                    _anno[k] = copy.deepcopy(v)
                _frame_annos.append(_anno)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in _frame_annos
                if obj.get("iscrowd", 0) == 0
            ]
            sorted_annos = [_get_dummy_anno(self.num_classes) for _ in range(len(ids))]

            for _anno in annos:
                idx = ids[_anno["id"]]
                sorted_annos[idx] = _anno
            _gt_ids = [_anno["id"] for _anno in sorted_annos]

            instances = utils.annotations_to_instances(sorted_annos, image_shape, mask_format="bitmask")
            if not self.is_tgt:
                instances.gt_classes = torch.tensor(
                    [self.src2tgt[c] if c in self.src2tgt else -1 for c in instances.gt_classes.tolist()]
                )
            instances.gt_ids = torch.tensor(_gt_ids)
            instances = filter_empty_instances(instances)
            # if instances.has("gt_masks"):
            #     instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            #     instances = filter_empty_instances(instances)
            if not instances.has("gt_masks"):
                instances.gt_masks = BitMasks(torch.empty((0, *image_shape)))
            dataset_dict["instances"].append(instances)

        return dataset_dict


class CocoClipDatasetMapper:
    """
    A callable which takes a COCO image which converts into multiple frames,
    and map it into a format used by the model.
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        is_tgt: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        copy_paste_augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        sampling_frame_num: int = 2,
        sampling_frame_range: int = 5,
        src_dataset_name: str = "",
        tgt_dataset_name: str = "",
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        # fmt: off
        self.is_train                = is_train
        self.is_tgt                  = is_tgt
        self.augmentations           = T.AugmentationList(augmentations)
        self.cp_augmentations        = T.AugmentationList(copy_paste_augmentations)
        self.image_format            = image_format
        self.sampling_frame_num      = sampling_frame_num
        self.sampling_frame_range    = sampling_frame_range
        self.copy_paste_augmentation = bool(copy_paste_augmentations)

        if not is_tgt:
            self.src_metadata = MetadataCatalog.get(src_dataset_name)
            self.tgt_metadata = MetadataCatalog.get(tgt_dataset_name)
            if tgt_dataset_name.startswith("ytvis_2019"):
                src2tgt = COCO_TO_YTVIS_2019
            elif tgt_dataset_name.startswith("ytvis_2021"):
                src2tgt = COCO_TO_YTVIS_2021
            elif tgt_dataset_name.startswith("ovis"):
                src2tgt = COCO_TO_OVIS
            else:
                raise NotImplementedError

            self.src2tgt = {}
            for k, v in src2tgt.items():
                self.src2tgt[
                    self.src_metadata.thing_dataset_id_to_contiguous_id[k]
                ] = self.tgt_metadata.thing_dataset_id_to_contiguous_id[v]

        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = True, is_tgt: bool = True):
        if is_tgt:
            augs = build_augmentation(cfg, is_train)
        else:
            augs = build_pseudo_augmentation(cfg, is_train)

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        copy_paste_augmentation = cfg.INPUT.PSEUDO.COPY_PASTE

        if not is_tgt and copy_paste_augmentation:
            cp_augs = build_pseudo_augmentation(cfg, is_train)
        else:
            cp_augs = []

        ret = {
            "is_train": is_train,
            "is_tgt": is_tgt,
            "augmentations": augs,
            "copy_paste_augmentations": cp_augs,
            "image_format": cfg.INPUT.FORMAT,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "tgt_dataset_name": cfg.DATASETS.TRAIN[-1],
        }

        return ret

    def __call__(self, dataset_dict, paste_dataset_dict=None):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        img_annos = dataset_dict.pop("annotations", None)
        file_name = dataset_dict.pop("file_name", None)
        original_image = utils.read_image(file_name, format=self.image_format)

        if self.is_train and self.copy_paste_augmentation:
            paste_dataset_dict = copy.deepcopy(paste_dataset_dict)

            paste_img_annos = paste_dataset_dict.pop("annotations", None)
            paste_file_name = paste_dataset_dict.pop("file_name", None)
            paste_original_image = utils.read_image(paste_file_name, format=self.image_format)

        if self.is_train:
            video_length = random.randrange(16, 49)
            ref_frame = random.randrange(video_length)

            start_idx = max(0, ref_frame-self.sampling_frame_range)
            end_idx = min(video_length, ref_frame+self.sampling_frame_range + 1)

            selected_idx = np.random.choice(
                np.array(list(range(start_idx, ref_frame)) + list(range(ref_frame+1, end_idx))),
                self.sampling_frame_num - 1,
            )
            selected_idx = selected_idx.tolist() + [ref_frame]
            selected_idx = sorted(selected_idx)
        else:
            video_length = self.sampling_frame_num
            selected_idx = list(range(self.sampling_frame_num))

        dataset_dict["video_len"] = video_length
        dataset_dict["frame_idx"] = selected_idx
        dataset_dict["image"] = []
        dataset_dict["instances"] = []
        dataset_dict["file_names"] = [file_name] * self.sampling_frame_num
        for _ in range(self.sampling_frame_num):
            utils.check_image_size(dataset_dict, original_image)

            image, instances = self.make_annos_to_instances(original_image, img_annos, self.augmentations)

            if instances is None:
                dataset_dict["image"].append(image)
                continue
            if self.is_train and self.copy_paste_augmentation:
                paste_image, paste_instances = self.make_annos_to_instances(paste_original_image, paste_img_annos, self.cp_augmentations)
                if paste_image is not None:
                    h_origin, w_origin = image.shape[1:]
                    resize_ratio = random.uniform(0.5, 1.0)
                    h_new = int(h_origin * resize_ratio)
                    w_new = int(w_origin * resize_ratio)

                    h_shift = random.randint(0, h_origin - h_new)
                    w_shift = random.randint(0, w_origin - w_new)

                    resized_paste_image = F.interpolate(paste_image.unsqueeze(0).float(), size=(h_new, w_new), mode="bilinear", align_corners=False).squeeze(0).byte()
                    resized_paste_masks = F.interpolate(paste_instances.gt_masks.float().unsqueeze(0), size=(h_new, w_new), mode="bilinear", align_corners=False).squeeze(0).bool()

                    _, mask_w, mask_h = instances.gt_masks.size()
                    masks_new_all = torch.zeros(len(paste_instances), mask_w, mask_h, dtype=torch.uint8)
                    images_new_all = torch.zeros_like(image)

                    images_new_all[:, h_shift:h_shift+h_new, w_shift:w_shift+w_new] += resized_paste_image
                    masks_new_all[:, h_shift:h_shift+h_new, w_shift:w_shift+w_new] += resized_paste_masks

                    paste_image = images_new_all.byte()
                    paste_masks = masks_new_all.bool()

                    iou_matrix = self.mask_iou_matrix(paste_masks, instances.gt_masks, mode='ioy')
                    keep = (iou_matrix.max(1)[0] < 0.5) & (paste_instances.gt_ids != -1)

                    alpha = paste_masks[keep].sum(0) > 0
                    composited_image = image * (~alpha) + paste_image * alpha

                    if keep.any():
                        instances.gt_masks = instances.gt_masks * ~alpha[None]

                    paste_instances.gt_ids += len(instances)
                    paste_instances.gt_ids[~keep] = -1
                    paste_instances.gt_masks = paste_masks
                    paste_instances._image_size = (h_origin, w_origin)

                    image = composited_image
                    instances = Instances.cat([instances, paste_instances])
                    
            dataset_dict["image"].append(image)
            dataset_dict["instances"].append(instances)

        return dataset_dict
    
    def make_annos_to_instances(self, original_image, img_annos, augmentations):
        aug_input = T.AugInput(original_image)
        transforms = augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if (img_annos is None) or (not self.is_train):
            return image, None

        _img_annos = []
        for anno in img_annos:
            _anno = {}
            for k, v in anno.items():
                _anno[k] = copy.deepcopy(v)
            _img_annos.append(_anno)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in _img_annos
            if obj.get("iscrowd", 0) == 0
        ]
        _gt_ids = list(range(len(annos)))
        for idx in range(len(annos)):
            if len(annos[idx]["segmentation"]) == 0:
                annos[idx]["segmentation"] = [np.array([0.0] * 6)]

        instances = utils.annotations_to_instances(annos, image_shape)
        if not self.is_tgt:
            instances.gt_classes = torch.tensor(
                [self.src2tgt[c] if c in self.src2tgt else -1 for c in instances.gt_classes.tolist()]
            )
        instances.gt_ids = torch.tensor(_gt_ids)
        # instances.gt_boxes = instances.gt_masks.get_bounding_boxes()  # NOTE we don't need boxes
        instances = filter_empty_instances(instances)
        h, w = instances.image_size
        if hasattr(instances, 'gt_masks'):
            gt_masks = instances.gt_masks
            gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            instances.gt_masks = gt_masks
        else:
            instances.gt_masks = torch.zeros((0, h, w), dtype=torch.uint8)
        
        return image, instances
    
    def mask_iou_matrix(self, x, y, mode='iou'):
            x = x.reshape(x.shape[0], -1).float() 
            y = y.reshape(y.shape[0], -1).float()
            inter_matrix = x @ y.transpose(1, 0) # n1xn2
            sum_x = x.sum(1)[:, None].expand(x.shape[0], y.shape[0])
            sum_y = y.sum(1)[None, :].expand(x.shape[0], y.shape[0])
            if mode == 'ioy':
                iou_matrix = inter_matrix / (sum_y) # [1, 1]
            else:
                iou_matrix = inter_matrix / (sum_x + sum_y - inter_matrix) # [1, 1]
            return iou_matrix
    
class CopyPasteMapDataset(MapDataset):
    def __init__(self, dataset, map_func):
        super().__init__(dataset, map_func)
        self.dataset = dataset
        self.map_func = map_func

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)
        sample_idx = random.randint(0, len(self._dataset) - 1)

        while True:
            data = self._map_func(self._dataset[cur_idx], self._dataset[sample_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]
            sample_idx = random.randint(0, len(self._dataset) - 1)

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )
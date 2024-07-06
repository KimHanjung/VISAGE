# VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement
[Hanjung Kim](http://kimhanjung.github.io), Jaehyun Kang, [Miran Heo](https://sites.google.com/view/miranheo), [Sukjun Hwang](https://sukjunhwang.github.io), [Seoung Wug Oh](https://sites.google.com/view/seoungwugoh), [Seon Joo Kim](https://sites.google.com/site/seonjookim/)

[[`arXiv`](https://arxiv.org/abs/2312.04885)] [[`Project`](https://kimhanjung.github.io/VISAGE/)] [[`BibTeX`](#CitingVISAGE)]

<div align="center">
  <img src="https://kimhanjung.github.io/images/visage.png" width="75%" height="75%"/>
</div>

### Features
* Video Instance Segmentation by leveraging an appearance information.
* Support major video instance segmentation datasets: YouTubeVIS 2019/2021/2022, Occluded VIS (OVIS).

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

For dataset preparation instructions, refer to [Preparing Datasets for VISAGE](datasets/README.md).

We provide a script `train_net_video.py`, that is made to train all the configs provided in VISAGE.

To train a model with "train_net_video.py", first setup the corresponding datasets following
[datasets/README.md](./datasets/README.md), then download the COCO pre-trained instance segmentation weights ([R50](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl), [Swin-L](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl)) and put them in the current working directory.
Once these are set up, run:
```
python train_net_video.py --num-gpus 4 \
  --config-file configs/youtubevis_2019/visage_R50_bs16.yaml
```

To evaluate a model's performance, use
```
python train_net_video.py \
  --config-file configs/youtubevis_2019/visage_R50_bs16.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```

## Model Zoo

### YouTube-VIS 2019

| Name                                      | Backbone             | AP   | AP50 | AP75 | AR1  | AR10 | Link                                                                                                             |
| ------------------------------------------ | -------------------- | ---- | ---- | ---- | ---- | ---- | ---------------------------------------------------------------------------------------------------------------- |
| VISAGE | ResNet-50            | 55.1 | 78.1 | 60.6 | 51.0 | 62.3 | [model](https://drive.google.com/file/d/142wqVU_Gz28L_Yaq6j3NlxAZLSf4E7d4/view?usp=sharing)|

### YouTube-VIS 2021

| Name                                      | Backbone             | AP   | AP50 | AP75 | AR1  | AR10 | Link                                                                                                             |
| ------------------------------------------ | -------------------- | ---- | ---- | ---- | ---- | ---- | ---------------------------------------------------------------------------------------------------------------- |
| VISAGE | ResNet-50            | 51.6 | 73.8 | 56.1 | 43.6 | 59.3 | [model](https://drive.google.com/file/d/1RCVCYOiEsgym9Mb7MMvTX-74rsKdSLRF/view?usp=sharing)|

### OVIS

| Name                                      | Backbone             | AP   | AP50 | AP75 | AR1  | AR10 | Link                                                                                                             |
| ------------------------------------------ | -------------------- | ---- | ---- | ---- | ---- | ---- | ---------------------------------------------------------------------------------------------------------------- |
| VISAGE | ResNet-50            | 36.2 | 60.3 | 35.3 | 17.0 | 40.3 | [model](https://drive.google.com/file/d/1L73nGcKjsZz8XH7Bx0UtbkE4m9-vf6Iq/view?usp=sharing)|


## License

The majority of VISAGE is licensed under a
[Apache-2.0 License](LICENSE).
However portions of the project are available under separate license terms: Detectron2([Apache-2.0 License](https://github.com/facebookresearch/detectron2/blob/main/LICENSE)), IFC([Apache-2.0 License](https://github.com/sukjunhwang/IFC/blob/master/LICENSE)), Mask2Former([MIT License](https://github.com/facebookresearch/Mask2Former/blob/main/LICENSE)), Deformable-DETR([Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE)), MinVIS([Nvidia Source Code License-NC](https://github.com/NVlabs/MinVIS/blob/main/LICENSE)),and VITA([Apache-2.0 License](https://github.com/sukjunhwang/VITA/blob/main/LICENSE)).


## <a name="CitingVISAGE"></a>Citing VISAGE

```BibTeX
@misc{kim2024visage,
      title={VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement}, 
      author={Hanjung Kim and Jaehyun Kang and Miran Heo and Sukjun Hwang and Seoung Wug Oh and Seon Joo Kim},
      year={2024},
      eprint={2312.04885},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

This repo is largely based on Mask2Former (https://github.com/facebookresearch/Mask2Former) and MinVIS (https://github.com/NVlabs/MinVIS) and VITA (https://github.com/sukjunhwang/VITA).

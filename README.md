# POST: Panoptic Segmentation and Tracking

## Architecture
<img width="2054" alt="architecture" src="https://github.com/user-attachments/assets/adbcfe36-7b6e-441e-b5b7-4e40a5da25b8">

## Abstract
Abstract
The long-standing goal of computer vision is to gain a high-level understanding of digital images and videos, similar to how the human brain is able to perceive objects, movement and depth in it’s visual field.
Recently advances in Convolutional Neural Networks have pushed computer vision to the limit. By dividing the task of human-like visual perception into bite-sized challenges such as classification, segmentation and tracking and establishing benchmarks significant progress was made, even outperforming humans in certain areas. However, there is still a long way to go in building a unified model, that performs all these tasks at once and is also capable of working robustly in a real world scenario and not only on a given dataset.
The proposed robust one-stage segmentation and tracking model aims to further this quest, by unifying the tasks of panoptic segmentation and tracking. Furthermore, our goal for this model is not to be bound to any specific benchmark dataset, but to provide robustness on real world examples.
We accomplish this goal by extending Panoptic-DeepLab [Che+20] by an previous offset branch, to enable it to track objects in a video. Furthermore, we train this model on multiple datasets simultaneously without setting hyperparameters for any specific dataset.

## Installation
Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).

## Training

To train a model with run:
```bash
cd /path/to/detectron2/projects/Post
python train_net.py --config-file configs/KITTI-MOTS/post_R_52_os16_mg124_poly_200k_bs1_kitti_mots_crop_384_dsconv.yaml
```

## Inference

Model evaluation can be done similarly:
```bash
cd /path/to/detectron2/projects/Post
python train_net.py --config-file configs/KITTI-MOTS/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_kitti_mots_dsconv.yaml --inference-only MODEL.WEIGHTS models/model_final_5e6da2.pkl INPUT.CROP.ENABLED False
```
## Benchmark network speed

If you want to benchmark the network speed without post-processing, you can run the evaluation script with `MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED True`:
```bash
cd /path/to/detectron2/projects/Poar
python train_net.py --config-file configs/KITTI-MOTS/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_kitti_mots_dsconv.yaml --eval-only MODEL.WEIGHTS /path/to/model_checkpoint MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED True
```

## Cityscapes Panoptic Segmentation
Cityscapes models are trained with ImageNet pretraining.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Output<br/>resolution</th>
<th valign="bottom">PQ</th>
<th valign="bottom">SQ</th>
<th valign="bottom">RQ</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">AP</th>
<th valign="bottom">Memory (M)</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left">Panoptic-DeepLab</td>
<td align="center">R50-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 58.6 </td>
<td align="center"> 80.9 </td>
<td align="center"> 71.2 </td>
<td align="center"> 75.9 </td>
<td align="center"> 29.8 </td>
<td align="center"> 8668 </td>
<td align="center"> - </td>
<td align="center">model&nbsp;|&nbsp;metrics</td>
</tr>
 <tr><td align="left"><a href="configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024.yaml">Panoptic-DeepLab</a></td>
<td align="center">R52-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 60.3 </td>
<td align="center"> 81.5 </td>
<td align="center"> 72.9 </td>
<td align="center"> 78.2 </td>
<td align="center"> 33.2 </td>
<td align="center"> 9682 </td>
<td align="center"> 30841561 </td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/model_final_bd324a.pkl
">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32/metrics.json
">metrics</a></td>
</tr>
 <tr><td align="left"><a href="configs/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv.yaml">Panoptic-DeepLab (DSConv)</a></td>
<td align="center">R52-DC5</td>
<td align="center">1024&times;2048</td>
<td align="center"> 60.3 </td>
<td align="center"> 81.0 </td>
<td align="center"> 73.2 </td>
<td align="center"> 78.7 </td>
<td align="center"> 32.1 </td>
<td align="center"> 10466 </td>
<td align="center"> 33148034 </td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv/model_final_23d03a.pkl
">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/Cityscapes-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_90k_bs32_crop_512_1024_dsconv/metrics.json
">metrics</a></td>
</tr>
</tbody></table>

Note:
- [R52](https://dl.fbaipublicfiles.com/detectron2/DeepLab/R-52.pkl): a ResNet-50 with its first 7x7 convolution replaced by 3 3x3 convolutions. This modification has been used in most semantic segmentation papers. We pre-train this backbone on ImageNet using the default recipe of [pytorch examples](https://github.com/pytorch/examples/tree/master/imagenet).
- DC5 means using dilated convolution in `res5`.
- We use a smaller training crop size (512x1024) than the original paper (1025x2049), we find using larger crop size (1024x2048) could further improve PQ by 1.5% but also degrades AP by 3%.
- The implementation with regular Conv2d in ASPP and head is much heavier head than the original paper.
- This implementation does not include optimized post-processing code needed for deployment. Post-processing the network
  outputs now takes similar amount of time to the network itself. Please refer to speed in the
  original paper for comparison.
- DSConv refers to using DepthwiseSeparableConv2d in ASPP and decoder. The implementation with DSConv is identical to the original paper.

## COCO Panoptic Segmentation
COCO models are trained with ImageNet pretraining on 16 V100s.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Method</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Output<br/>resolution</th>
<th valign="bottom">PQ</th>
<th valign="bottom">SQ</th>
<th valign="bottom">RQ</th>
<th valign="bottom">Box AP</th>
<th valign="bottom">Mask AP</th>
<th valign="bottom">Memory (M)</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml">Panoptic-DeepLab (DSConv)</a></td>
<td align="center">R52-DC5</td>
<td align="center">640&times;640</td>
<td align="center"> 35.5 </td>
<td align="center"> 77.3 </td>
<td align="center"> 44.7 </td>
<td align="center"> 18.6 </td>
<td align="center"> 19.7 </td>
<td align="center">  </td>
<td align="center"> 246448865 </td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv/model_final_5e6da2.pkl
">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv/metrics.json
">metrics</a></td>
</tr>
</tbody></table>

Note:
- [R52](https://dl.fbaipublicfiles.com/detectron2/DeepLab/R-52.pkl): a ResNet-50 with its first 7x7 convolution replaced by 3 3x3 convolutions. This modification has been used in most semantic segmentation papers. We pre-train this backbone on ImageNet using the default recipe of [pytorch examples](https://github.com/pytorch/examples/tree/master/imagenet).
- DC5 means using dilated convolution in `res5`.
- This reproduced number matches the original paper (35.5 vs. 35.1 PQ).
- This implementation does not include optimized post-processing code needed for deployment. Post-processing the network
<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>

### What's New
* It is powered by the [PyTorch](https://pytorch.org) deep learning framework.
* Includes more features such as panoptic segmentation, Densepose, Cascade R-CNN, rotated bounding boxes, PointRend,
  DeepLab, etc.
* Can be used as a library to support [different projects](projects/) on top of it.
  We'll open source more research projects in this way.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).
* Models can be exported to TorchScript format or Caffe2 format for deployment.

See our [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

See [INSTALL.md](INSTALL.md).

## Getting Started

Follow the [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html) to
install detectron2.

See [Getting Started with Detectron2](https://detectron2.readthedocs.io/tutorials/getting_started.html),
and the [Colab Notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
to learn about basic usage.

Learn more at our [documentation](https://detectron2.readthedocs.org).
And see [projects/](projects/) for some projects that are built on top of detectron2.

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).


## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

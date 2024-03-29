# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Callable, Dict, List, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, DepthwiseSeparableConv2d, ShapeSpec, get_norm
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.projects.deeplab import DeepLabV3PlusHead
from detectron2.projects.deeplab.loss import DeepLabCE
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry

from .post_processing import PostProcessing

__all__ = ["Post", "INS_EMBED_BRANCHES_REGISTRY", "build_ins_embed_branch"]


INS_EMBED_BRANCHES_REGISTRY = Registry("INS_EMBED_BRANCHES")
INS_EMBED_BRANCHES_REGISTRY.__doc__ = """
Registry for instance embedding branches, which make instance embedding
predictions from feature maps.
"""


@META_ARCH_REGISTRY.register()
class Post(nn.Module):
    """
    Main class for panoptic segmentation and tracking architectures.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())
        self.ins_embed_head = build_ins_embed_branch(cfg, self.backbone.output_shape())
        self.prev_offset_head = build_prev_offset_head(cfg, self.backbone.output_shape())
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.stuff_area = cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA
        self.threshold = cfg.MODEL.PANOPTIC_DEEPLAB.CENTER_THRESHOLD
        self.nms_kernel = cfg.MODEL.PANOPTIC_DEEPLAB.NMS_KERNEL
        self.top_k = cfg.MODEL.PANOPTIC_DEEPLAB.TOP_K_INSTANCE
        self.predict_instances = cfg.MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES
        self.post_processing = PostProcessing(
            thing_ids=self.meta.thing_dataset_id_to_contiguous_id.values(),
            label_divisor=self.meta.label_divisor,
            stuff_area=self.stuff_area,
            void_label=-1,
            predict_instances=self.predict_instances,
            threshold=self.threshold,
            nms_kernel=self.nms_kernel,
            top_k=self.top_k,
        )
        self.use_depthwise_separable_conv = cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        assert (
            cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV
            == cfg.MODEL.PANOPTIC_DEEPLAB.USE_DEPTHWISE_SEPARABLE_CONV
        )
        self.size_divisibility = cfg.MODEL.PANOPTIC_DEEPLAB.SIZE_DIVISIBILITY
        self.benchmark_network_speed = cfg.MODEL.PANOPTIC_DEEPLAB.BENCHMARK_NETWORK_SPEED

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
                   * "prev_image": Tensor, previous image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * "center": center points heatmap ground truth
                   * "offset": pixel offsets to center points ground truth
                   * "prev_offset": pixel offsets from previous masks to center points ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """
        # To avoid error in ASPP layer when input has different size.
        size_divisibility = (
            self.size_divisibility
            if self.size_divisibility > 0
            else self.backbone.size_divisibility
        )

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, size_divisibility)
        # Features: the output of the backbone for frame t
        features = self.backbone(images.tensor)
        
        if "prev_image" in batched_inputs[0]:
            prev_images = [x["prev_image"].to(self.device) for x in batched_inputs]
            prev_images = [(x - self.pixel_mean) / self.pixel_std for x in prev_images]
            prev_images = ImageList.from_tensors(prev_images, size_divisibility)
            # Previous features: output of the backbone for frame t-1
            prev_features = self.backbone(prev_images.tensor)

        losses = {}
        if "sem_seg" in batched_inputs[0] and self.training:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
            targets = ImageList.from_tensors(
                targets, size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
            if "sem_seg_weights" in batched_inputs[0]:
                # The default D2 DatasetMapper may not contain "sem_seg_weights"
                # Avoid error in testing when default DatasetMapper is used.
                weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                weights = ImageList.from_tensors(weights, size_divisibility).tensor
            else:
                weights = None
        else:
            targets = None
            weights = None

        # Check if there are targets or running inference
        if targets is not None or not self.training:
            sem_seg_results, sem_seg_losses = self.sem_seg_head(features, targets, weights)
            losses.update(sem_seg_losses)

        if "center" in batched_inputs[0] and "offset" in batched_inputs[0] and self.training:
            center_targets = [x["center"].to(self.device) for x in batched_inputs]
            center_targets = ImageList.from_tensors(
                center_targets, size_divisibility
            ).tensor.unsqueeze(1)
            center_weights = [x["center_weights"].to(self.device) for x in batched_inputs]
            center_weights = ImageList.from_tensors(center_weights, size_divisibility).tensor

            offset_targets = [x["offset"].to(self.device) for x in batched_inputs]
            offset_targets = ImageList.from_tensors(offset_targets, size_divisibility).tensor
            offset_weights = [x["offset_weights"].to(self.device) for x in batched_inputs]
            offset_weights = ImageList.from_tensors(offset_weights, size_divisibility).tensor
        else:
            center_targets = None
            center_weights = None

            offset_targets = None
            offset_weights = None

        center_results, offset_results, center_losses, offset_losses = self.ins_embed_head(
            features, center_targets, center_weights, offset_targets, offset_weights
        )
        losses.update(center_losses)
        losses.update(offset_losses)

        if 'prev_offset' in batched_inputs[0] and self.training:
            prev_offset_targets = [x["prev_offset"].to(self.device) for x in batched_inputs]
            prev_offset_targets = ImageList.from_tensors(prev_offset_targets, size_divisibility).tensor
            prev_offset_weights = [x["prev_offset_weights"].to(self.device) for x in batched_inputs]
            prev_offset_weights = ImageList.from_tensors(prev_offset_weights, size_divisibility).tensor
        else:
            prev_offset_targets = None
            prev_offset_weights = None        

        # Check if there are targets or running inference
        if prev_offset_targets is not None or not self.training:
            if "prev_image" in batched_inputs[0]:  
                # Concatenate the output layer of the backbone for the previous and current image
                prev_features['res5'] = torch.cat([features['res5'], prev_features['res5']], dim=1)
                # Calls the 'forward' function of 'prev_offset_head'  
                # in training results are None and in inference losses are empty {}
                prev_offset_results, prev_offset_losses = self.prev_offset_head(
                    prev_features, prev_offset_targets, prev_offset_weights
                )
                losses.update(prev_offset_losses)
            else:
                prev_offset_results = [None]

        if self.training:
            return losses

        if self.benchmark_network_speed:
            return []
        
        # Post processing
        processed_results = []
        results = zip(sem_seg_results, center_results, offset_results, prev_offset_results, batched_inputs, images.image_sizes)
        for result in results:
            processed_result = self.post_processing(*result)
            processed_results.append(processed_result)
        return processed_results


@SEM_SEG_HEADS_REGISTRY.register()
class PanopticDeepLabSemSegHead(DeepLabV3PlusHead):
    """
    A semantic segmentation head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        loss_weight: float,
        loss_type: str,
        loss_top_k: float,
        ignore_value: int,
        num_classes: int,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "in_features"
                (each element in "in_features" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
        """
        super().__init__(
            input_shape,
            decoder_channels=decoder_channels,
            norm=norm,
            ignore_value=ignore_value,
            **kwargs,
        )
        assert self.decoder_only

        self.loss_weight = loss_weight
        use_bias = norm == ""
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.head[0])
            weight_init.c2_xavier_fill(self.head[1])
        self.predictor = Conv2d(head_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        if loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_value)
        elif loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=ignore_value, top_k_percent_pixels=loss_top_k)
        else:
            raise ValueError("Unexpected loss type: %s" % loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        ret["loss_top_k"] = cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K
        return ret

    def forward(self, features, targets=None, weights=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)
        if self.training:
            return None, self.losses(y, targets, weights)
        else:
            y = F.interpolate(
                y, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            return y, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        y = self.head(y)
        y = self.predictor(y)
        return y

    def losses(self, predictions, targets, weights=None):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.loss(predictions, targets, weights)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


def build_ins_embed_branch(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.INS_EMBED_HEAD.NAME
    return INS_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape)


@INS_EMBED_BRANCHES_REGISTRY.register()
class PanopticDeepLabInsEmbedHead(DeepLabV3PlusHead):
    """
    A instance embedding head described in :paper:`Panoptic-DeepLab`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        center_loss_weight: float,
        offset_loss_weight: float,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "in_features"
                (each element in "in_features" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center point prediction.
            offset_loss_weight (float): loss weight for center offset prediction.
        """
        super().__init__(input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs)
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        use_bias = norm == ""
        # center prediction
        # `head` is additional transform before predictor
        self.center_head = nn.Sequential(
            Conv2d(
                decoder_channels[0],
                decoder_channels[0],
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, decoder_channels[0]),
                activation=F.relu,
            ),
            Conv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=3,
                padding=1,
                bias=use_bias,
                norm=get_norm(norm, head_channels),
                activation=F.relu,
            ),
        )
        weight_init.c2_xavier_fill(self.center_head[0])
        weight_init.c2_xavier_fill(self.center_head[1])
        self.center_predictor = Conv2d(head_channels, 1, kernel_size=1)
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        # offset prediction
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.offset_head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.offset_head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.offset_head[0])
            weight_init.c2_xavier_fill(self.offset_head[1])
        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.INS_EMBED_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.INS_EMBED_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape=input_shape,
            in_features=cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES,
            project_channels=cfg.MODEL.INS_EMBED_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.INS_EMBED_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.INS_EMBED_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.INS_EMBED_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            offset_loss_weight=cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def forward(
        self,
        features,
        center_targets=None,
        center_weights=None,
        offset_targets=None,
        offset_weights=None,
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        center, offset = self.layers(features)
        if self.training:
            return (
                None,
                None,
                self.center_losses(center, center_targets, center_weights),
                self.offset_losses(offset, offset_targets, offset_weights),
            )
        else:
            center = F.interpolate(
                center, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            offset = (
                F.interpolate(
                    offset, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                * self.common_stride
            )
            return center, offset, {}, {}

    def layers(self, features):
        assert self.decoder_only
        y = super().layers(features)
        # center
        center = self.center_head(y)
        center = self.center_predictor(center)
        # offset
        offset = self.offset_head(y)
        offset = self.offset_predictor(offset)
        return center, offset

    def center_losses(self, predictions, targets, weights):
        predictions = F.interpolate(
            predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
        )
        loss = self.center_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_center": loss * self.center_loss_weight}
        return losses

    def offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(
                predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            * self.common_stride
        )
        loss = self.offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_offset": loss * self.offset_loss_weight}
        return losses

def build_prev_offset_head(cfg, input_shape):
    """
    Build a instance embedding branch from `cfg.MODEL.PREV_OFFSET_HEAD.NAME`.
    """
    name = cfg.MODEL.PREV_OFFSET_HEAD.NAME
    return INS_EMBED_BRANCHES_REGISTRY.get(name)(cfg, input_shape, prev_offset=True)

@INS_EMBED_BRANCHES_REGISTRY.register()
class PrevOffsetHead(DeepLabV3PlusHead):
    """
    Instance offset of the previous frame to the current center
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        decoder_channels: List[int],
        norm: Union[str, Callable],
        head_channels: int,
        prev_offset_loss_weight: float,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "in_features"
                (each element in "in_features" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            offset_loss_weight (float): loss weight for center offset prediction.
        """
        super().__init__(input_shape, decoder_channels=decoder_channels, norm=norm, **kwargs)
        # 'decoder_only' should be false because 'num_classes' was not passed
        assert self.decoder_only

        self.prev_offset_loss_weight = prev_offset_loss_weight
        use_bias = norm == ""
        
        # prev_offset prediction
        # `head` is additional transform before predictor
        if self.use_depthwise_separable_conv:
            # We use a single 5x5 DepthwiseSeparableConv2d to replace
            # 2 3x3 Conv2d since they have the same receptive field.
            self.prev_offset_head = DepthwiseSeparableConv2d(
                decoder_channels[0],
                head_channels,
                kernel_size=5,
                padding=2,
                norm1=norm,
                activation1=F.relu,
                norm2=norm,
                activation2=F.relu,
            )
        else:
            self.prev_offset_head = nn.Sequential(
                Conv2d(
                    decoder_channels[0],
                    decoder_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[0]),
                    activation=F.relu,
                ),
                Conv2d(
                    decoder_channels[0],
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, head_channels),
                    activation=F.relu,
                ),
            )
            weight_init.c2_xavier_fill(self.prev_offset_head[0])
            weight_init.c2_xavier_fill(self.prev_offset_head[1])
        self.prev_offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        nn.init.normal_(self.prev_offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.prev_offset_predictor.bias, 0)

        self.prev_offset_loss = nn.L1Loss(reduction="none")

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.PREV_OFFSET_HEAD.CONVS_DIM] * (
            len(cfg.MODEL.PREV_OFFSET_HEAD.IN_FEATURES) - 1
        ) + [cfg.MODEL.PREV_OFFSET_HEAD.ASPP_CHANNELS]
        ret = dict(
            input_shape=input_shape,
            in_features=cfg.MODEL.PREV_OFFSET_HEAD.IN_FEATURES,
            project_channels=cfg.MODEL.PREV_OFFSET_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.PREV_OFFSET_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.PREV_OFFSET_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.PREV_OFFSET_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.PREV_OFFSET_HEAD.NORM,
            train_size=train_size,
            head_channels=cfg.MODEL.PREV_OFFSET_HEAD.HEAD_CHANNELS,
            prev_offset_loss_weight=cfg.MODEL.PREV_OFFSET_HEAD.PREV_OFFSET_LOSS_WEIGHT,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def forward(
        self,
        features,
        prev_offset_targets=None,
        prev_offset_weights=None,
    ):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        prev_offset = self.this_layers(features)
        if self.training:
            return (
                None,
                self.prev_offset_losses(prev_offset, prev_offset_targets, prev_offset_weights),
            )
        else:
            prev_offset = (
                F.interpolate(
                    prev_offset, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                * self.common_stride
            )
            return prev_offset, {}

    def this_layers(self, features):
        # Reverse feature maps into top-down order (from low to high resolution)
        for f in self.in_features[::-1]:
            x = features[f]
            proj_x = self.decoder[f]["project_conv"](x)
            if self.decoder[f]["fuse_conv"] is None:
                # This is aspp module
                y = proj_x
            else:
                # Upsample y
                y = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
                y = torch.cat([proj_x, y], dim=1)
                y = self.decoder[f]["fuse_conv"](y)

        prev_offset = self.prev_offset_head(y)
        prev_offset = self.prev_offset_predictor(prev_offset)
        return prev_offset

    def layers(self, features):

        # TODO: 
        y = super().layers(features)
        # prev_offset
        prev_offset = self.prev_offset_head(y)
        prev_offset = self.prev_offset_predictor(prev_offset)
        return prev_offset

    def prev_offset_losses(self, predictions, targets, weights):
        predictions = (
            F.interpolate(
                predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
            )
            * self.common_stride
        )
        loss = self.prev_offset_loss(predictions, targets) * weights
        if weights.sum() > 0:
            loss = loss.sum() / weights.sum()
        else:
            loss = loss.sum() * 0
        losses = {"loss_prev_offset": loss * self.prev_offset_loss_weight}
        return losses
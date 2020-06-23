import inspect

import cv2
import albumentations
import mmcv
import numpy as np

from albumentations import Compose
from imagecorruptions import corrupt
from numpy import random
from glob import glob

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..registry import PIPELINES


@PIPELINES.register_module
class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                masks = [
                    mmcv.imrescale(
                        mask, results['scale_factor'], interpolation='nearest')
                    for mask in results[key]
                ]
            else:
                mask_size = (results['img_shape'][1], results['img_shape'][0])
                masks = [
                    mmcv.imresize(mask, mask_size, interpolation='nearest')
                    for mask in results[key]
                ]
            results[key] = masks

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


@PIPELINES.register_module
class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        w = img_shape[1]
        flipped = bboxes.copy()
        flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
        flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        return flipped

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [mask[:, ::-1] for mask in results[key]]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


###
@PIPELINES.register_module
class RandomVerticalFlip(object):
    """Flip the image & bbox & mask vertically.

    If the input dict contains the key "vertical_flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None):
        self.flip_ratio = flip_ratio
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1

    def bbox_flip(self, bboxes, img_shape):
        """Flip bboxes vertically.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        h = img_shape[0]
        flipped = bboxes.copy()
        flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
        flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        return flipped

    def __call__(self, results):
        if 'vertical_flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['vertical_flip'] = flip
        if results['vertical_flip']:
            # flip image
            results['img'] = mmcv.imflip(results['img'], direction='vertical')
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = [mask[::-1, :] for mask in results[key]]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)



@PIPELINES.register_module
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = mmcv.impad(results['img'], self.size)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            padded_masks = [
                mmcv.impad(mask, pad_shape, pad_val=self.pad_val)
                for mask in results[key]
            ]
            results[key] = np.stack(padded_masks, axis=0)

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


@PIPELINES.register_module
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str


@PIPELINES.register_module
class RandomCrop(object):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        img = results['img']
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

        # filter out the gt bboxes that are completely cropped
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                gt_bboxes[:, 3] > gt_bboxes[:, 1])
            # if no gt bbox remains after cropping, just skip this image
            if not np.any(valid_inds):
                return None
            results['gt_bboxes'] = gt_bboxes[valid_inds, :]
            if 'gt_labels' in results:
                results['gt_labels'] = results['gt_labels'][valid_inds]

            # filter and crop the masks
            if 'gt_masks' in results:
                valid_gt_masks = []
                for i in np.where(valid_inds)[0]:
                    gt_mask = results['gt_masks'][i][crop_y1:crop_y2, crop_x1:
                                                     crop_x2]
                    valid_gt_masks.append(gt_mask)
                results['gt_masks'] = valid_gt_masks

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={})'.format(
            self.crop_size)


@PIPELINES.register_module
class SegResizeFlipPadRescale(object):
    """A sequential transforms to semantic segmentation maps.

    The same pipeline as input images is applied to the semantic segmentation
    map, and finally rescale it by some scale factor. The transforms include:
    1. resize
    2. flip
    3. pad
    4. rescale (so that the final size can be different from the image size)

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        if results['keep_ratio']:
            gt_seg = mmcv.imrescale(
                results['gt_semantic_seg'],
                results['scale'],
                interpolation='nearest')
        else:
            gt_seg = mmcv.imresize(
                results['gt_semantic_seg'],
                results['scale'],
                interpolation='nearest')
        if results['flip']:
            gt_seg = mmcv.imflip(gt_seg)
        if gt_seg.shape != results['pad_shape']:
            gt_seg = mmcv.impad(gt_seg, results['pad_shape'][:2])
        if self.scale_factor != 1:
            gt_seg = mmcv.imrescale(
                gt_seg, self.scale_factor, interpolation='nearest')
        results['gt_semantic_seg'] = gt_seg
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(scale_factor={})'.format(
            self.scale_factor)


@PIPELINES.register_module
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        img = results['img']
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(brightness_delta={}, contrast_range={}, '
                     'saturation_range={}, hue_delta={})').format(
                         self.brightness_delta, self.contrast_range,
                         self.saturation_range, self.hue_delta)
        return repr_str


@PIPELINES.register_module
class Expand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 to_rgb=True,
                 ratio_range=(1, 4),
                 seg_ignore_label=None,
                 prob=0.5):
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob

    def __call__(self, results):
        if random.uniform(0, 1) > self.prob:
            return results

        img, boxes = [results[k] for k in ('img', 'gt_bboxes')]

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        boxes = boxes + np.tile((left, top), 2).astype(boxes.dtype)

        results['img'] = expand_img
        results['gt_bboxes'] = boxes

        if 'gt_masks' in results:
            expand_gt_masks = []
            for mask in results['gt_masks']:
                expand_mask = np.full((int(h * ratio), int(w * ratio)),
                                      0).astype(mask.dtype)
                expand_mask[top:top + h, left:left + w] = mask
                expand_gt_masks.append(expand_mask)
            results['gt_masks'] = expand_gt_masks

        # not tested
        if 'gt_semantic_seg' in results:
            assert self.seg_ignore_label is not None
            gt_seg = results['gt_semantic_seg']
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label).astype(gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results['gt_semantic_seg'] = expand_gt_seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, to_rgb={}, ratio_range={}, ' \
                    'seg_ignore_label={})'.format(
                        self.mean, self.to_rgb, self.ratio_range,
                        self.seg_ignore_label)
        return repr_str


@PIPELINES.register_module
class MinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, results):
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) *
                        (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                results['img'] = img
                results['gt_bboxes'] = boxes
                results['gt_labels'] = labels

                if 'gt_masks' in results:
                    valid_masks = [
                        results['gt_masks'][i] for i in range(len(mask))
                        if mask[i]
                    ]
                    results['gt_masks'] = [
                        gt_mask[patch[1]:patch[3], patch[0]:patch[2]]
                        for gt_mask in valid_masks
                    ]

                # not tested
                if 'gt_semantic_seg' in results:
                    results['gt_semantic_seg'] = results['gt_semantic_seg'][
                        patch[1]:patch[3], patch[0]:patch[2]]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(min_ious={}, min_crop_size={})'.format(
            self.min_ious, self.min_crop_size)
        return repr_str


@PIPELINES.register_module
class Corrupt(object):

    def __init__(self, corruption, severity=1):
        self.corruption = corruption
        self.severity = severity

    def __call__(self, results):
        results['img'] = corrupt(
            results['img'].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(corruption={}, severity={})'.format(
            self.corruption, self.severity)
        return repr_str


@PIPELINES.register_module
class Albu(object):

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        """
        Adds custom transformations from Albumentations lib.
        Please, visit `https://albumentations.readthedocs.io`
        to get more information.

        transforms (list): list of albu transformations
        bbox_params (dict): bbox_params for albumentation `Compose`
        keymap (dict): contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): whether to skip the image
                                      if no ann left after aug
        """

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        Inherits some of `build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                'type must be a str or valid type, but got {}'.format(
                    type(obj_type)))

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """
        Dictionary mapper.
        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)

            # filter label_fields
            if self.filter_lost_elements:

                results['idx_mapper'] = np.arange(len(results['bboxes']))

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = [
                        results['masks'][i] for i in results['idx_mapper']
                    ]

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transformations={})'.format(self.transformations)
        return repr_str


###
@PIPELINES.register_module
class AlbuMine(object):

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        """
        Adds custom transformations from Albumentations lib.
        Please, visit `https://albumentations.readthedocs.io`
        to get more information.

        transforms (list): list of albu transformations
        bbox_params (dict): bbox_params for albumentation `Compose`
        keymap (dict): contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): whether to skip the image
                                      if no ann left after aug
        """
        """###
            While the gt label is empty, do aug again until the gt label is not empty.
        """

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        Inherits some of `build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                'type must be a str or valid type, but got {}'.format(
                    type(obj_type)))

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """
        Dictionary mapper.
        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        while True:
            # print("Before: ", results['gt_labels'].shape)
            results_cur = self.aug(**results)
            # print("After: ", np.array(results_cur['gt_labels']).shape)
            if len(results_cur['gt_labels']) > 0:
                results = results_cur
                break


        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)

            # filter label_fields
            if self.filter_lost_elements:

                results['idx_mapper'] = np.arange(len(results['bboxes']))

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = [
                        results['masks'][i] for i in results['idx_mapper']
                    ]

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transformations={})'.format(self.transformations)
        return repr_str





###
@PIPELINES.register_module
class GtBoxBasedCrop(object):
    """
        Crop around the gt bbox.
        Note:
            Here 'img_shape' is change to the shape of img_cropped.
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size # (w, h)

    def __call__(self, results):

        img = results['img']
        gt_bboxes = results['gt_bboxes']
        gt_labels = results['gt_labels']


        img_cropped, gt_bboxes_cropped, gt_labels_cropped = \
            self._crop_patch(img, gt_bboxes, gt_labels)

        results['img'] = img_cropped
        results['gt_bboxes'] = gt_bboxes_cropped
        results['gt_labels'] = gt_labels_cropped
        results['img_shape'] = img_cropped.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(crop_size={})'.format(self.crop_size)
        return repr_str


    def _crop_patch(self, img, gt_bboxes, gt_labels):

        H, W, C = img.shape
        px, py = self.crop_size

        if px > W or py > H:
            pad_w, pad_h = 0, 0

            if px > W:
                pad_w = px - W
                W = px
            if py > H:
                pad_h = py - H
                H = py

            img = cv2.copyMakeBorder(img, 0, int(pad_h), 0, int(pad_w), cv2.BORDER_CONSTANT, 0)


        obj_num = gt_bboxes.shape[0]
        select_gt_id = np.random.randint(0, obj_num)
        x1, y1, x2, y2 = gt_bboxes[select_gt_id, :]

        if x2-x1>px:
            nx = np.random.randint(x1, x2 - px + 1)
        else:
            nx = np.random.randint(max(x2 - px, 0), min(x1 + 1, W - px + 1))
        
        if y2-y1>py:
            ny = np.random.randint(y1, y2 - py + 1)
        else:
            ny = np.random.randint(max(y2 - py, 0), min(y1 + 1, H - py + 1))


        patch_coord = np.zeros((1, 4), dtype="int")
        patch_coord[0, 0] = nx
        patch_coord[0, 1] = ny
        patch_coord[0, 2] = nx + px
        patch_coord[0, 3] = ny + py 

        index = self._compute_overlap(patch_coord, gt_bboxes, 0.5)
        index = np.squeeze(index, axis=0)
        index[select_gt_id] = True

        patch = img[ny: ny + py, nx: nx + px, :]
        gt_bboxes = gt_bboxes[index, :]
        gt_labels = gt_labels[index]

        gt_bboxes[:, 0] = np.maximum(gt_bboxes[:, 0] - patch_coord[0, 0], 0)
        gt_bboxes[:, 1] = np.maximum(gt_bboxes[:, 1] - patch_coord[0, 1], 0)
        gt_bboxes[:, 2] = np.minimum(gt_bboxes[:, 2] - patch_coord[0, 0], px - 1)
        gt_bboxes[:, 3] = np.minimum(gt_bboxes[:, 3] - patch_coord[0, 1], py - 1)

        return patch, gt_bboxes, gt_labels



    def _compute_overlap(self, a, b, over_threshold=0.5):
        """
        Parameters
        ----------
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
        Returns
        -------
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
        """
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
        ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

        iw = np.maximum(iw, 0)
        ih = np.maximum(ih, 0)

        # ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih
        ua = area

        ua = np.maximum(ua, np.finfo(float).eps)

        intersection = iw * ih

        overlap = intersection / ua
        index = overlap > over_threshold
        return index



@PIPELINES.register_module
class Bboxes_Jitter(object):
    def __init__(self,shift_ratio=0.1):
        self.shift_ratio = shift_ratio
    def __call__(self,results):
        img,gt_bboxes = results['img'],results['gt_bboxes']
        h,w,c = img.shape
        
        # box_w,box_h
        box_w,box_h = gt_bboxes[:,2] - gt_bboxes[:,0],gt_bboxes[:,3] - gt_bboxes[:,1]
        
        left =   np.random.uniform(box_w*(-self.shift_ratio),box_w*self.shift_ratio)
        top =    np.random.uniform(box_h*(-self.shift_ratio),box_h*self.shift_ratio)
        right =  np.random.uniform(box_w*(-self.shift_ratio),box_w*self.shift_ratio)
        buttom = np.random.uniform(box_h*(-self.shift_ratio),box_h*self.shift_ratio)
        
        gt_bboxes[:,0] = np.maximum(0,gt_bboxes[:,0] + left).astype(int)
        gt_bboxes[:,1] = np.maximum(0,gt_bboxes[:,1] + top).astype(int)
        gt_bboxes[:,2] = np.minimum(w-1,gt_bboxes[:,2] + right).astype(int)
        gt_bboxes[:,3] = np.minimum(h-1,gt_bboxes[:,3] + buttom).astype(int)
        results['gt_bboxes'] = gt_bboxes
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str +="(shift_ratio={})".format(self.shift_ratio)
        return repr_str



@PIPELINES.register_module
class CopyPasting(object):
    def __init__(self,iou_threshold=0.05,num_copy=16,drop_rate=0.2,crop_size=(1600,1600), drop_small_roi=False, \
        copy_pasting_dict=r"/home/admin/jupyter/Datasets/copy_pasting_dict.npz", \
        background_dir=r"/home/admin/jupyter/Datasets/neg_rois/"):
        # drop_rate : 以一定概率丢弃当前图，重新加载阴性图像
        self.num_copy = num_copy
        copy_pasting_dict = np.load(copy_pasting_dict)
        self.pos_list = copy_pasting_dict['pos']
        self.candida_list = copy_pasting_dict['candida']
        self.trichomonas_list = copy_pasting_dict['trichomonas']
        self.background_list = glob(background_dir+"*.npz")
        
        self.iou_threshold = iou_threshold
        self.gt_crop = GtBoxBasedCrop(crop_size)
        self.drop_rate = drop_rate
        self.drop_small_roi = drop_small_roi
        self.crop_size = crop_size
        
        
    def _check_box_overlap(self,box1,box_list):
        len_box_list = len(box_list)
        b1_x1,b1_y1,b1_x2,b1_y2 = box1
        b_x1_list,b_y1_list,b_x2_list,b_y2_list = box_list[:,0],box_list[:,1],box_list[:,2],box_list[:,3]

        inter_rect_x1 = np.maximum(b1_x1,b_x1_list)
        inter_rect_y1 = np.maximum(b1_y1,b_y1_list)
        inter_rect_x2 = np.minimum(b1_x2,b_x2_list)
        inter_rect_y2 = np.minimum(b1_y2,b_y2_list)
        inter_area = np.maximum(inter_rect_x2-inter_rect_x1, 0)*np.maximum(inter_rect_y2-inter_rect_y1, 0)
        
        box1_area = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        box_area_list = np.maximum(b_x2_list-b_x1_list,0)*np.maximum(b_y2_list-b_y1_list,0)
        iou_list = inter_area/(box1_area+box_area_list-inter_area)

        # if len(np.where(box1_area+box_area_list-inter_area==0)[0]) > 0:
        #     print(box1_area, box_area_list[np.where(box1_area+box_area_list-inter_area==0)[0]], inter_area[np.where(box1_area+box_area_list-inter_area==0)[0]])

        if np.max(iou_list)>self.iou_threshold:
            return False
        else:
            return True


    def __call__(self,results):
        if np.random.uniform(0,1) < self.drop_rate:
            npz = np.load(self.background_list[np.random.randint(0,len(self.background_list))])
            image,label = npz['img'],npz['label']
            
            results_neg = {'img':image,'gt_labels':label[:,-1].reshape(-1),'gt_bboxes':label[:,:4]}
            results_neg = self.gt_crop(results_neg)

            results['img'] = results_neg['img']
            results['gt_labels'] = results_neg['gt_labels']
            results['gt_bboxes'] = results_neg['gt_bboxes']
            results['ori_shape'] = image.shape

        elif self.drop_small_roi:
            if results['ori_shape'][0] < self.crop_size[1] or results['ori_shape'][1] < self.crop_size[0]:  # shape: (h, w, c) size: (x, y)
                return results

        # if results['ori_shape'][0] < self.crop_size[0] or results['ori_shape'][1] < self.crop_size[1]:
        #     print(results['ori_shape'])

        image,gt_labels,gt_bboxes = results['img'],results['gt_labels'],results['gt_bboxes']
        h,w,c = image.shape
        mask = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        # print(gt_labels)
        if 5 in gt_labels:    # candida
            copylist_indice = np.random.randint(0,len(self.candida_list),self.num_copy) # random select
            copylist = self.candida_list[copylist_indice]

        elif 6 in gt_labels: # Tri
            copylist_indice = np.random.randint(0,len(self.trichomonas_list),self.num_copy) # random select
            copylist = self.trichomonas_list[copylist_indice]
        elif -1 in gt_labels:
            random_choice = np.random.choice([self.candida_list,self.pos_list,self.trichomonas_list])
            copylist_indice = np.random.randint(0,len(random_choice),self.num_copy)
            copylist = random_choice[copylist_indice]
        else:               # pos
            copylist_indice = np.random.randint(0,len(self.pos_list),self.num_copy)
            copylist = self.pos_list[copylist_indice]
        cnt = 0
        for i in range(len(copylist)):
            copy_box = np.load(copylist[i])
            copy_image,copy_label = copy_box['img'],copy_box['label']
            copy_image_h,copy_image_w = copy_image.shape[:2]
            for attempt in range(10):
                if copy_image_h >= h or copy_image_w >= w:
                    break
                top = np.random.randint(0,h-copy_image_h)
                left = np.random.randint(0,w-copy_image_w)
                box1 = np.array([left,top,left+copy_image_w,top+copy_image_h])
                check_overlap_flag = self._check_box_overlap(box1,gt_bboxes)
                if check_overlap_flag:
                    image[top:top+copy_image_h,left:left+copy_image_w,:] = copy_image
                    gt_bboxes = np.concatenate((gt_bboxes,box1[None,:]),axis=0)
                    gt_labels = np.concatenate((gt_labels,copy_label.reshape(1)),axis=0)
                    cnt += 1
                    break
        image = cv2.inpaint(image,mask,inpaintRadius=3,flags=cv2.INPAINT_NS)
        results['img'] = image
#         print(len(gt_bboxes))
        gt_bboxes = gt_bboxes[gt_labels!=-1]
        gt_labels = gt_labels[gt_labels!=-1]
#         print(len(gt_bboxes))
        results['gt_labels'] = gt_labels
        results['gt_bboxes'] = gt_bboxes
#         print(cnt)
        return results



@PIPELINES.register_module
class CopyPastingPseudoCandida(object):
    def __init__(self, pasting_rate=1., iou_threshold=0.05, \
        overlap_threshold=0.1, num_copy=8, score_thred=0.1, label_offset=4, \
        pseudo_dict=r"/home/admin/jupyter/Datasets/pseudo_label/pseudo_label_dict_candida.npz", \
        gt_dict=r"/home/admin/jupyter/Datasets/copy_pasting_dict.npz"):

        """
            Paste pseudo bboxed to the img.
            - input:
                score_thred: float. The score thred of pseudo label. (0.6, 0.3, 0.1)
        """

        self.pasting_rate = pasting_rate
        self.num_copy = num_copy
        pseudo_dict = np.load(pseudo_dict)
        gt_dict = np.load(gt_dict)
        self.pseudo_list = np.array(pseudo_dict['npz_dirs'].item()[str(score_thred)])
        self.gt_list = np.array(gt_dict['candida'])

        # import pdb
        # pdb.set_trace()

        self.bbox_list = np.concatenate([self.pseudo_list, self.gt_list]) # combine both pseudo and gt bbox

        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        self.label_offset = label_offset
                

    def __call__(self,results):
        if np.random.uniform(0,1) < self.pasting_rate:
            
            image,gt_labels,gt_bboxes,ori_shape = \
                results['img'],results['gt_labels'],results['gt_bboxes'],results['ori_shape']

            h,w,c = image.shape

            mask = np.zeros((h, w, 1), np.uint8)


            copylist_indice = np.random.randint(0,len(self.bbox_list),self.num_copy) # random select
            copylist = self.bbox_list[copylist_indice]

            h = np.min([h, ori_shape[0]])
            w = np.min([w, ori_shape[1]])

            cnt = 0
            for i in range(len(copylist)):
                copy_box = np.load(copylist[i])
                copy_image,copy_label = copy_box['img'],copy_box['label']

                if self.label_offset > 0:
                    copy_label -= self.label_offset

                copy_image_h,copy_image_w = copy_image.shape[:2]
                for attempt in range(10):
                    if copy_image_h >= h or copy_image_w >= w:
                        break
                    top = np.random.randint(0,h-copy_image_h)
                    left = np.random.randint(0,w-copy_image_w)
                    box1 = np.array([left,top,left+copy_image_w,top+copy_image_h])
                    check_overlap_flag = self._check_box_overlap(box1,gt_bboxes)
                    if check_overlap_flag:
                        image[top:top+copy_image_h,left:left+copy_image_w,:] = copy_image
                        gt_bboxes = np.concatenate((gt_bboxes,box1[None,:]),axis=0)
                        gt_labels = np.concatenate((gt_labels,copy_label.reshape(1)),axis=0)

                        mask = cv2.rectangle(mask, (left, top), (left+copy_image_w, top+copy_image_h), 255, 6)

                        cnt += 1
                        break

            image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

            results['img'] = image
            results['gt_labels'] = gt_labels
            results['gt_bboxes'] = gt_bboxes

        return results

    def _check_box_overlap(self,box1,box_list):
        len_box_list = len(box_list)
        
        if len_box_list == 0:
            return True
        
        b1_x1,b1_y1,b1_x2,b1_y2 = box1
        b_x1_list,b_y1_list,b_x2_list,b_y2_list = box_list[:,0],box_list[:,1],box_list[:,2],box_list[:,3]

        inter_rect_x1 = np.maximum(b1_x1,b_x1_list)
        inter_rect_y1 = np.maximum(b1_y1,b_y1_list)
        inter_rect_x2 = np.minimum(b1_x2,b_x2_list)
        inter_rect_y2 = np.minimum(b1_y2,b_y2_list)
        inter_area = np.maximum(inter_rect_x2-inter_rect_x1, 0)*np.maximum(inter_rect_y2-inter_rect_y1, 0)
        
        box1_area = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        box_area_list = np.maximum(b_x2_list-b_x1_list,0)*np.maximum(b_y2_list-b_y1_list,0)
        iou_list = inter_area/(box1_area+box_area_list-inter_area)
        
        box1_overlap = inter_area / box1_area
        box_list_overlap = inter_area / box_area_list
        
#         pdb.set_trace()
        # print(box1_overlap, box_list_overlap)
        
        if np.max(iou_list)>self.iou_threshold:
            return False
        if np.max(box1_overlap) > self.overlap_threshold:
            return False
        if np.max(box_list_overlap)>self.overlap_threshold:
            return False
        
        return True





@PIPELINES.register_module
class ReplaceBackground(object):
    def __init__(self, drop_rate=0.2, crop_size=(1600,1600), \
        background_dir=r"/home/admin/jupyter/Datasets/neg_rois/"):

        self.background_list = glob(background_dir+"*.npz")
        self.gt_crop = GtBoxBasedCrop(crop_size)
        self.drop_rate = drop_rate
        self.crop_size = crop_size

    def __call__(self,results):

        if np.random.uniform(0,1) < self.drop_rate:
            npz = np.load(self.background_list[np.random.randint(0,len(self.background_list))])
            image,label = npz['img'],npz['label']
            
            results_neg = {'img':image,'gt_labels':label[:,-1].reshape(-1),'gt_bboxes':label[:,:4]}
            results_neg = self.gt_crop(results_neg)

            # results['img'] = results_neg['img']

            src_img = results['img']
            bg_img = results_neg['img']

            # mask = cv2.cvtColor(bg_img, cv2.COLOR_RGB2GRAY)

            mask = np.zeros((bg_img.shape[0], bg_img.shape[1], 1), np.uint8)

            for i in range(len(results['gt_bboxes'])):

                bg_img[int(results['gt_bboxes'][i][1]): int(results['gt_bboxes'][i][3]), \
                        int(results['gt_bboxes'][i][0]): int(results['gt_bboxes'][i][2]), :] = \
                src_img[int(results['gt_bboxes'][i][1]): int(results['gt_bboxes'][i][3]), \
                        int(results['gt_bboxes'][i][0]): int(results['gt_bboxes'][i][2]), :]

                mask = cv2.rectangle(mask,(int(results['gt_bboxes'][i][0]),int(results['gt_bboxes'][i][1])), \
                                    (int(results['gt_bboxes'][i][2]), int(results['gt_bboxes'][i][3])), 255, 6)

            image = cv2.inpaint(bg_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

            results['img'] = image

        return results



@PIPELINES.register_module
class ReplaceBackgroundCandida(object):
    def __init__(self, drop_rate=0.85, crop_size=(3000,3000), iou_threshold=0.05, overlap_threshold=0.1, \
        mix_up_dict=r"/home/admin/jupyter/Datasets/mix_up_dict.npz"):
        """
            - inputs:
                drop_small_roi: whether to skip the roi smaller than crop_size.
                iou_threshold: the IoU thredshold to keep the gt_bbox.
                overlap_thred: the overlap thredshold to keep the gt_bbox.
        """

        mix_up_dict = np.load(mix_up_dict)
        
        self.background_list = mix_up_dict['background'].tolist() + mix_up_dict['pos'].tolist() + mix_up_dict['Trichomonas'].tolist()

        self.gt_crop = GtBoxBasedCrop(crop_size)
        self.drop_rate = drop_rate
        self.crop_size = crop_size
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        
    def __call__(self,results):

        if np.random.uniform(0,1) < self.drop_rate:
            
            npz = np.load(self.background_list[np.random.randint(0,len(self.background_list))])
            image,label = npz['img'],npz['label']
            
            results_neg = {'img':image,'gt_labels':label[:,-1].reshape(-1),'gt_bboxes':label[:,:4], 'ori_shape': image.shape}
            results_neg = self.gt_crop(results_neg)


            src_img = results['img']
            bg_img = results_neg['img']
            gt_bbox_list = []
            
            gt_bboxes = np.empty(shape=[0, 4], dtype = np.int32)
            gt_labels = np.empty(shape=[0], dtype = np.int32)

            mask = np.zeros((bg_img.shape[0], bg_img.shape[1], 1), np.uint8)

            for i in range(len(results['gt_bboxes'])): # save the gt bboxes as a list
                
                gt_bbox_list.append(
                    src_img[int(results['gt_bboxes'][i][1]): int(results['gt_bboxes'][i][3]), \
                            int(results['gt_bboxes'][i][0]): int(results['gt_bboxes'][i][2]), :].copy()
                )


            cnt = 0
            
            h = np.min([self.crop_size[1], results_neg['ori_shape'][0]])
            w = np.min([self.crop_size[0], results_neg['ori_shape'][1]])
            
#             print(self.crop_size, results_neg['ori_shape'])
            
            
            for i in range(len(gt_bbox_list)):
                copy_image = gt_bbox_list[i]
                copy_image_h,copy_image_w = copy_image.shape[:2]
                for attempt in range(10):
                    if copy_image_h >= h or copy_image_w >= w:
                        break
                    top = np.random.randint(0,h-copy_image_h)
                    left = np.random.randint(0,w-copy_image_w)
                    box1 = np.array([left,top,left+copy_image_w,top+copy_image_h])
                    check_overlap_flag = self._check_box_overlap(box1,gt_bboxes)
                    if check_overlap_flag:
                        bg_img[top:top+copy_image_h,left:left+copy_image_w,:] = copy_image
                        gt_bboxes = np.concatenate((gt_bboxes,box1[None,:]),axis=0)
                        gt_labels = np.concatenate((gt_labels,results['gt_labels'][i].reshape(1)),axis=0)
                        
                        mask = cv2.rectangle(mask, (left, top), (left+copy_image_w, top+copy_image_h), 255, 6)
                        
                        cnt += 1
                        break    
            
            if cnt == 0: # No gt_bbox was pasted.
                return results
            
            image = cv2.inpaint(bg_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

            results['img'] = image
            results['gt_bboxes'] = gt_bboxes
            results['gt_labels'] = gt_labels

        return results

    def _check_box_overlap(self,box1,box_list):
        len_box_list = len(box_list)
        
        if len_box_list == 0:
            return True
        
        b1_x1,b1_y1,b1_x2,b1_y2 = box1
        b_x1_list,b_y1_list,b_x2_list,b_y2_list = box_list[:,0],box_list[:,1],box_list[:,2],box_list[:,3]

        inter_rect_x1 = np.maximum(b1_x1,b_x1_list)
        inter_rect_y1 = np.maximum(b1_y1,b_y1_list)
        inter_rect_x2 = np.minimum(b1_x2,b_x2_list)
        inter_rect_y2 = np.minimum(b1_y2,b_y2_list)
        inter_area = np.maximum(inter_rect_x2-inter_rect_x1, 0)*np.maximum(inter_rect_y2-inter_rect_y1, 0)
        
        box1_area = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        box_area_list = np.maximum(b_x2_list-b_x1_list,0)*np.maximum(b_y2_list-b_y1_list,0)
        iou_list = inter_area/(box1_area+box_area_list-inter_area)
        
        box1_overlap = inter_area / box1_area
        box_list_overlap = inter_area / box_area_list
        
#         pdb.set_trace()
        # print(box1_overlap, box_list_overlap)
        
        if np.max(iou_list)>self.iou_threshold:
            return False
        if np.max(box1_overlap) > self.overlap_threshold:
            return False
        if np.max(box_list_overlap)>self.overlap_threshold:
            return False
        
        return True
         



@PIPELINES.register_module
class ReplaceBackgroundSubCls(object):
    def __init__(self, drop_rate=0.85, crop_size=(3000,3000), iou_threshold=0.05, overlap_threshold=0.1, \
        cls_type='Candida', mix_up_dict=r"/home/admin/jupyter/Datasets/mix_up_dict.npz"):

        mix_up_dict = np.load(mix_up_dict)
        
        if cls_type == 'Candida':
            self.background_list = mix_up_dict['background'].tolist() + mix_up_dict['pos'].tolist() + mix_up_dict['Trichomonas'].tolist()
        elif cls_type == 'Trichomonas':
            self.background_list = mix_up_dict['background'].tolist() + mix_up_dict['pos'].tolist() + mix_up_dict['Candida'].tolist()
        else:
            raise

        self.gt_crop = GtBoxBasedCrop(crop_size)
        self.drop_rate = drop_rate
        self.crop_size = crop_size
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        
    def __call__(self,results):

        if np.random.uniform(0,1) < self.drop_rate:
            
            npz = np.load(self.background_list[np.random.randint(0,len(self.background_list))])
            image,label = npz['img'],npz['label']
            
            results_neg = {'img':image,'gt_labels':label[:,-1].reshape(-1),'gt_bboxes':label[:,:4], 'ori_shape': image.shape}
            results_neg = self.gt_crop(results_neg)


            src_img = results['img']
            bg_img = results_neg['img']
            gt_bbox_list = []
            
            gt_bboxes = np.empty(shape=[0, 4], dtype = np.int32)
            gt_labels = np.empty(shape=[0], dtype = np.int32)

            mask = np.zeros((bg_img.shape[0], bg_img.shape[1], 1), np.uint8)

            for i in range(len(results['gt_bboxes'])): # save the gt bboxes as a list
                
                gt_bbox_list.append(
                    src_img[int(results['gt_bboxes'][i][1]): int(results['gt_bboxes'][i][3]), \
                            int(results['gt_bboxes'][i][0]): int(results['gt_bboxes'][i][2]), :].copy()
                )


            cnt = 0
            
            h = np.min([self.crop_size[1], results_neg['ori_shape'][0]])
            w = np.min([self.crop_size[0], results_neg['ori_shape'][1]])
            
#             print(self.crop_size, results_neg['ori_shape'])
            
            
            for i in range(len(gt_bbox_list)):
                copy_image = gt_bbox_list[i]
                copy_image_h,copy_image_w = copy_image.shape[:2]
                for attempt in range(10):
                    if copy_image_h >= h or copy_image_w >= w:
                        break
                    top = np.random.randint(0,h-copy_image_h)
                    left = np.random.randint(0,w-copy_image_w)
                    box1 = np.array([left,top,left+copy_image_w,top+copy_image_h])
                    check_overlap_flag = self._check_box_overlap(box1,gt_bboxes)
                    if check_overlap_flag:
                        bg_img[top:top+copy_image_h,left:left+copy_image_w,:] = copy_image
                        gt_bboxes = np.concatenate((gt_bboxes,box1[None,:]),axis=0)
                        gt_labels = np.concatenate((gt_labels,results['gt_labels'][i].reshape(1)),axis=0)
                        
                        mask = cv2.rectangle(mask, (left, top), (left+copy_image_w, top+copy_image_h), 255, 6)
                        
                        cnt += 1
                        break    
            
            if cnt == 0: # No gt_bbox was pasted.
                return results
            
            image = cv2.inpaint(bg_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

            results['img'] = image
            results['gt_bboxes'] = gt_bboxes
            results['gt_labels'] = gt_labels

        return results

    def _check_box_overlap(self,box1,box_list):
        len_box_list = len(box_list)
        
        if len_box_list == 0:
            return True
        
        b1_x1,b1_y1,b1_x2,b1_y2 = box1
        b_x1_list,b_y1_list,b_x2_list,b_y2_list = box_list[:,0],box_list[:,1],box_list[:,2],box_list[:,3]

        inter_rect_x1 = np.maximum(b1_x1,b_x1_list)
        inter_rect_y1 = np.maximum(b1_y1,b_y1_list)
        inter_rect_x2 = np.minimum(b1_x2,b_x2_list)
        inter_rect_y2 = np.minimum(b1_y2,b_y2_list)
        inter_area = np.maximum(inter_rect_x2-inter_rect_x1, 0)*np.maximum(inter_rect_y2-inter_rect_y1, 0)
        
        box1_area = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        box_area_list = np.maximum(b_x2_list-b_x1_list,0)*np.maximum(b_y2_list-b_y1_list,0)
        iou_list = inter_area/(box1_area+box_area_list-inter_area)
        
        box1_overlap = inter_area / box1_area
        box_list_overlap = inter_area / box_area_list
        
#         pdb.set_trace()
        # print(box1_overlap, box_list_overlap)
        
        if np.max(iou_list)>self.iou_threshold:
            return False
        if np.max(box1_overlap) > self.overlap_threshold:
            return False
        if np.max(box_list_overlap)>self.overlap_threshold:
            return False
        
        return True
       

@PIPELINES.register_module
class ReplaceBackgroundWrtOriRatio(object):
    def __init__(self, crop_size=(3000,3000), iou_threshold=0.05, overlap_threshold=0.1, \
        cls_type='pos', mix_up_dict=r"/home/admin/jupyter/Datasets/mix_up_dict.npz"):
        """
            Randomly replace the bg according to the original number ratio.
            - inputs:
                cls_type: 'pos', 'Candida' or 'Trichomonas'.
                iou_threshold: the IoU thredshold to keep the gt_bbox.
                overlap_thred: the overlap thredshold to keep the gt_bbox.

            The number of each class to calculate ratio.
                - pos_image: 1440 (kfb)
                - neg_image: 250 (kfb)
                - image_num: 1690 (kfb)
                
                - pos_image:
                    - pos: 2953 (roi)
                    - Candida: 485 (roi)
                    - Trichomonas: 234 (roi)
                    - roi_num: 3670 (roi)


        """

        CLS_NUM = {
            'pos_image': 1440,
            'neg_image': 250,
            'pos': 2953,
            'Candida': 485,
            'Trichomonas': 234,
            'roi_num': 3670,
            'image_num': 1690,
        }

        mix_up_dict = np.load(mix_up_dict)
        
        if cls_type == 'pos':
            keep_rate = (CLS_NUM['pos'] / CLS_NUM['roi_num'] * CLS_NUM['pos_image']) / CLS_NUM['image_num']
            self.drop_rate = 1. - keep_rate
            self.use_neg_image_rate = CLS_NUM['neg_image'] / \
                (CLS_NUM['image_num'] - CLS_NUM['pos'] / CLS_NUM['roi_num'] * CLS_NUM['pos_image'])
            self.pos_background_list = mix_up_dict['Candida'].tolist() + mix_up_dict['Trichomonas'].tolist()
        elif cls_type == 'Candida':
            keep_rate = (CLS_NUM['Candida'] / CLS_NUM['roi_num'] * CLS_NUM['pos_image']) / CLS_NUM['image_num']
            self.drop_rate = 1. - keep_rate
            self.use_neg_image_rate = CLS_NUM['neg_image'] / \
                (CLS_NUM['image_num'] - CLS_NUM['Candida'] / CLS_NUM['roi_num'] * CLS_NUM['pos_image'])
            self.pos_background_list = mix_up_dict['pos'].tolist() + mix_up_dict['Trichomonas'].tolist()
        elif cls_type == 'Trichomonas':
            keep_rate = (CLS_NUM['Trichomonas'] / CLS_NUM['roi_num'] * CLS_NUM['pos_image']) / CLS_NUM['image_num']
            self.drop_rate = 1. - keep_rate
            self.use_neg_image_rate = CLS_NUM['neg_image'] / \
                (CLS_NUM['image_num'] - CLS_NUM['Trichomonas'] / CLS_NUM['roi_num'] * CLS_NUM['pos_image'])
            self.pos_background_list = mix_up_dict['pos'].tolist() + mix_up_dict['Candida'].tolist()

        else:
            raise

        self.neg_background_list = mix_up_dict['background'].tolist()

        # import pdb
        # pdb.set_trace()

        self.gt_crop = GtBoxBasedCrop(crop_size)
        self.crop_size = crop_size
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        
    def __call__(self,results):

        if np.random.uniform(0,1) < self.drop_rate: # replace background
            
            if np.random.uniform(0,1) < self.use_neg_image_rate:    # use neg image as background
                npz = np.load(self.neg_background_list[np.random.randint(0,len(self.neg_background_list))])
            else:   # use pos image as background
                npz = np.load(self.pos_background_list[np.random.randint(0,len(self.pos_background_list))])

            image,label = npz['img'],npz['label']
            
            results_neg = {'img':image,'gt_labels':label[:,-1].reshape(-1),'gt_bboxes':label[:,:4], 'ori_shape': image.shape}
            results_neg = self.gt_crop(results_neg)

            src_img = results['img']
            bg_img = results_neg['img']
            gt_bbox_list = []
            
            gt_bboxes = np.empty(shape=[0, 4], dtype = np.int32)
            gt_labels = np.empty(shape=[0], dtype = np.int32)

            mask = np.zeros((bg_img.shape[0], bg_img.shape[1], 1), np.uint8)

            for i in range(len(results['gt_bboxes'])): # save the gt bboxes as a list
                
                gt_bbox_list.append(
                    src_img[int(results['gt_bboxes'][i][1]): int(results['gt_bboxes'][i][3]), \
                            int(results['gt_bboxes'][i][0]): int(results['gt_bboxes'][i][2]), :].copy()
                )


            cnt = 0
            
            h = np.min([self.crop_size[1], results_neg['ori_shape'][0]])    # avoid paste the gt_bbox to the padding zeros areas.
            w = np.min([self.crop_size[0], results_neg['ori_shape'][1]])
            
#             print(self.crop_size, results_neg['ori_shape'])
            
            
            for i in range(len(gt_bbox_list)):
                copy_image = gt_bbox_list[i]
                copy_image_h,copy_image_w = copy_image.shape[:2]
                for attempt in range(10):
                    if copy_image_h >= h or copy_image_w >= w:
                        break
                    top = np.random.randint(0,h-copy_image_h)
                    left = np.random.randint(0,w-copy_image_w)
                    box1 = np.array([left,top,left+copy_image_w,top+copy_image_h])
                    check_overlap_flag = self._check_box_overlap(box1,gt_bboxes)
                    if check_overlap_flag:
                        bg_img[top:top+copy_image_h,left:left+copy_image_w,:] = copy_image
                        gt_bboxes = np.concatenate((gt_bboxes,box1[None,:]),axis=0)
                        gt_labels = np.concatenate((gt_labels,results['gt_labels'][i].reshape(1)),axis=0)
                        
                        mask = cv2.rectangle(mask, (left, top), (left+copy_image_w, top+copy_image_h), 255, 6)
                        
                        cnt += 1
                        break    
            
            if cnt == 0: # No gt_bbox was pasted.
                return results
            
            image = cv2.inpaint(bg_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

            results['img'] = image
            results['gt_bboxes'] = gt_bboxes
            results['gt_labels'] = gt_labels

        return results

    def _check_box_overlap(self,box1,box_list):
        len_box_list = len(box_list)
        
        if len_box_list == 0:
            return True
        
        b1_x1,b1_y1,b1_x2,b1_y2 = box1
        b_x1_list,b_y1_list,b_x2_list,b_y2_list = box_list[:,0],box_list[:,1],box_list[:,2],box_list[:,3]

        inter_rect_x1 = np.maximum(b1_x1,b_x1_list)
        inter_rect_y1 = np.maximum(b1_y1,b_y1_list)
        inter_rect_x2 = np.minimum(b1_x2,b_x2_list)
        inter_rect_y2 = np.minimum(b1_y2,b_y2_list)
        inter_area = np.maximum(inter_rect_x2-inter_rect_x1, 0)*np.maximum(inter_rect_y2-inter_rect_y1, 0)
        
        box1_area = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        box_area_list = np.maximum(b_x2_list-b_x1_list,0)*np.maximum(b_y2_list-b_y1_list,0)
        iou_list = inter_area/(box1_area+box_area_list-inter_area)
        
        box1_overlap = inter_area / box1_area
        box_list_overlap = inter_area / box_area_list
        
#         pdb.set_trace()
        # print(box1_overlap, box_list_overlap)
        
        if np.max(iou_list)>self.iou_threshold:
            return False
        if np.max(box1_overlap) > self.overlap_threshold:
            return False
        if np.max(box_list_overlap)>self.overlap_threshold:
            return False
        
        return True
      



@PIPELINES.register_module
class RandomShiftGtBBox(object):
    def __init__(self, shift_rate=0.5, iou_threshold=0.05, overlap_threshold=0.1):
        """
            Randomly shift the gt_bboxes.
            - inputs:
                shift_rate: the probability to shift a gt bbox.
                iou_threshold: the IoU thredshold to keep the gt_bbox.
                overlap_thred: the overlap thredshold to keep the gt_bbox.
        """


        self.shift_rate = shift_rate
        self.iou_threshold = iou_threshold
        self.overlap_threshold = overlap_threshold
        
    def __call__(self, results):

        shift_flag = np.random.rand(results['gt_bboxes'].shape[0]) < self.shift_rate

        shift_id = np.where(shift_flag)[0]
        keep_id = np.where(~shift_flag)[0]

        if len(shift_id) == 0:  # no shifting
            return results


        shift_gt_bboxes = results['gt_bboxes'][shift_id, :]
        keep_gt_bboxes = results['gt_bboxes'][keep_id, :]
        shift_gt_labels = results['gt_labels'][shift_id]
        keep_gt_labels = results['gt_labels'][keep_id]

        src_img = results['img']

        gt_bboxes = keep_gt_bboxes
        gt_labels = keep_gt_labels
        shift_gt_bbox_list = []


        mask = np.zeros((src_img.shape[0], src_img.shape[1], 1), np.uint8)
        for i in range(len(shift_gt_bboxes)):   # inpainting mask
            ltx = int(shift_gt_bboxes[i, 0])
            lty = int(shift_gt_bboxes[i, 1])
            brx = int(shift_gt_bboxes[i, 2])
            bry = int(shift_gt_bboxes[i, 3])

            # import pdb
            # pdb.set_trace()

            mask[lty: bry, ltx: brx, :] = 1

            shift_gt_bbox_list.append(
                src_img[lty: bry, ltx: brx, :].copy()
            )

        src_img = cv2.inpaint(src_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

        h = np.min([src_img.shape[0], results['ori_shape'][0]])
        w = np.min([src_img.shape[1], results['ori_shape'][1]])




        for i in range(len(shift_gt_bbox_list)):
            copy_image = shift_gt_bbox_list[i]
            copy_image_h,copy_image_w = copy_image.shape[:2]
            for attempt in range(10):
                if copy_image_h >= h or copy_image_w >= w:
                    break
                top = np.random.randint(0,h-copy_image_h)
                left = np.random.randint(0,w-copy_image_w)
                box1 = np.array([left,top,left+copy_image_w,top+copy_image_h])
                check_overlap_flag = self._check_box_overlap(box1,gt_bboxes)
                if check_overlap_flag:
                    src_img[top:top+copy_image_h,left:left+copy_image_w,:] = copy_image
                    gt_bboxes = np.concatenate((gt_bboxes,box1[None,:]),axis=0)
                    gt_labels = np.concatenate((gt_labels,shift_gt_labels[i].reshape(1)),axis=0)
                    
                    break   


        results['img'] = src_img
        results['gt_bboxes'] = gt_bboxes
        results['gt_labels'] = gt_labels

        return results


    def _check_box_overlap(self,box1,box_list):
        len_box_list = len(box_list)
        
        if len_box_list == 0:
            return True
        
        b1_x1,b1_y1,b1_x2,b1_y2 = box1
        b_x1_list,b_y1_list,b_x2_list,b_y2_list = box_list[:,0],box_list[:,1],box_list[:,2],box_list[:,3]

        inter_rect_x1 = np.maximum(b1_x1,b_x1_list)
        inter_rect_y1 = np.maximum(b1_y1,b_y1_list)
        inter_rect_x2 = np.minimum(b1_x2,b_x2_list)
        inter_rect_y2 = np.minimum(b1_y2,b_y2_list)
        inter_area = np.maximum(inter_rect_x2-inter_rect_x1, 0)*np.maximum(inter_rect_y2-inter_rect_y1, 0)
        
        box1_area = (b1_x2-b1_x1)*(b1_y2-b1_y1)
        box_area_list = np.maximum(b_x2_list-b_x1_list,0)*np.maximum(b_y2_list-b_y1_list,0)
        iou_list = inter_area/(box1_area+box_area_list-inter_area)
        
        box1_overlap = inter_area / box1_area
        box_list_overlap = inter_area / box_area_list
        
#         pdb.set_trace()
        # print(box1_overlap, box_list_overlap)
        
        if np.max(iou_list)>self.iou_threshold:
            return False
        if np.max(box1_overlap) > self.overlap_threshold:
            return False
        if np.max(box_list_overlap)>self.overlap_threshold:
            return False
        
        return True
      


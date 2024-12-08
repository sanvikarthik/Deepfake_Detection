"""
Video Face Manipulation Detection Through Ensemble of CNNs

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""
from pprint import pprint
from typing import Iterable, List, Tuple

import albumentations as A
import cv2
import numpy as np
import scipy
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms


def extract_meta_av(path: str) -> Tuple[int, int, int]:
    """
    Extract video height, width and number of frames to index the files.
    :param path: Path to the video file.
    :return: Tuple of (height, width, number of frames).
    """
    import av
    try:
        video = av.open(path)
        video_stream = video.streams.video[0]
        return video_stream.height, video_stream.width, video_stream.frames
    except (av.AVError, IndexError) as e:
        print(f"Error while processing file: {path}\n{e}")
        return 0, 0, 0


def extract_meta_cv(path: str) -> Tuple[int, int, int]:
    """
    Extract video height, width and number of frames using OpenCV.
    :param path: Path to the video file.
    :return: Tuple of (height, width, number of frames).
    """
    try:
        vid = cv2.VideoCapture(path)
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        return height, width, num_frames
    except Exception as e:
        print(f"Error while reading file: {path}\n{e}")
        return 0, 0, 0


def adapt_bb(frame_height: int, frame_width: int, bb_height: int, bb_width: int, left: int, top: int, right: int,
             bottom: int) -> Tuple[int, int, int, int]:
    """
    Adjust bounding box coordinates to fit within the frame dimensions.
    :param frame_height: Height of the video frame.
    :param frame_width: Width of the video frame.
    :param bb_height: Desired height of the bounding box.
    :param bb_width: Desired width of the bounding box.
    :param left, top, right, bottom: Original bounding box coordinates.
    :return: Adjusted bounding box coordinates (left, top, right, bottom).
    """
    x_ctr = (left + right) // 2
    y_ctr = (bottom + top) // 2
    new_top = max(y_ctr - bb_height // 2, 0)
    new_bottom = min(new_top + bb_height, frame_height)
    new_left = max(x_ctr - bb_width // 2, 0)
    new_right = min(new_left + bb_width, frame_width)
    return new_left, new_top, new_right, new_bottom


def extract_bb(frame: Image.Image, bb: Iterable[int], scale: str, size: int) -> Image.Image:
    """
    Extract a face from a frame using a bounding box and scaling policy.
    :param frame: Entire frame as a PIL.Image.
    :param bb: Bounding box (left, top, right, bottom).
    :param scale: Scaling policy ("scale", "crop", "tight").
    :param size: Size to scale the extracted face.
    :return: Cropped and resized face image.
    """
    left, top, right, bottom = bb
    if scale == "scale":
        bb_width = int(right) - int(left)
        bb_height = int(bottom) - int(top)
        bb_to_desired_ratio = min(size / bb_height, size / bb_width) if (bb_width > 0 and bb_height > 0) else 1.0
        bb_width = int(size / bb_to_desired_ratio)
        bb_height = int(size / bb_to_desired_ratio)
        left, top, right, bottom = adapt_bb(frame.height, frame.width, bb_height, bb_width, left, top, right, bottom)
        face = frame.crop((left, top, right, bottom)).resize((size, size), Image.BILINEAR)
    elif scale == "crop":
        left, top, right, bottom = adapt_bb(frame.height, frame.width, size, size, left, top, right, bottom)
        face = frame.crop((left, top, right, bottom))
    elif scale == "tight":
        left, top, right, bottom = adapt_bb(frame.height, frame.width, bottom - top, right - left, left, top, right,
                                            bottom)
        face = frame.crop((left, top, right, bottom))
    else:
        raise ValueError(f"Unknown scale value: {scale}")

    return face


def showimage(img_tensor: torch.Tensor):
    """
    Display an image from a PyTorch tensor.
    :param img_tensor: Image tensor with shape (C, H, W).
    """
    to_pil = transforms.ToPILImage()
    img = to_pil(img_tensor.cpu().detach())
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def make_train_tag(net_class: nn.Module,
                   face_policy: str,
                   patch_size: int,
                   traindb: List[str],
                   seed: int,
                   suffix: str,
                   debug: bool) -> str:
    """
    Create a unique training session tag.
    :param net_class: Neural network class.
    :param face_policy: Policy for face extraction.
    :param patch_size: Patch size.
    :param traindb: List of training databases.
    :param seed: Random seed.
    :param suffix: Additional suffix for the tag.
    :param debug: If true, prepend 'debug_' to the tag.
    :return: Training tag as a string.
    """
    tag_params = {
        "net": net_class.__name__,
        "traindb": '-'.join(traindb),
        "face": face_policy,
        "size": patch_size,
        "seed": seed
    }
    print("Parameters:")
    pprint(tag_params)
    tag = 'debug_' if debug else ''
    tag += '_'.join([f"{key}-{value}" for key, value in tag_params.items()])
    if suffix:
        tag += f"_{suffix}"
    print(f"Tag: {tag}")
    return tag




def get_transformer(face_policy: str, patch_size: int, net_normalizer: transforms.Normalize, train: bool):
    # Transformers and traindb
    if face_policy == 'scale':
        # The loader crops the face isotropically then scales to a square of size patch_size_load
        loading_transformations = [
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0,always_apply=True),
            A.Resize(height=patch_size,width=patch_size,always_apply=True),
        ]
        if train:
            downsample_train_transformations = [
                A.Downscale(scale_max=0.5, scale_min=0.5, p=0.5),  # replaces scaled dataset
            ]
        else:
            downsample_train_transformations = []
    elif face_policy == 'tight':
        # The loader crops the face tightly without any scaling
        loading_transformations = [
            A.LongestMaxSize(max_size=patch_size, always_apply=True),
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0,always_apply=True),
        ]
        if train:
            downsample_train_transformations = [
                A.Downscale(scale_max=0.5, scale_min=0.5, p=0.5),  # replaces scaled dataset
            ]
        else:
            downsample_train_transformations = []
    else:
        raise ValueError('Unknown value for face_policy: {}'.format(face_policy))

    if train:
        aug_transformations = [
            A.Compose([
                A.HorizontalFlip(),
                A.OneOf([
                    A.RandomBrightnessContrast(),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
                ]),
                A.OneOf([
                    A.ISONoise(),
                    A.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.03 * 255)),
                ]),
                A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR),
                A.ImageCompression(quality_lower=50, quality_upper=99),
            ], )
        ]
    else:
        aug_transformations = []

    # Common final transformations
    final_transformations = [
        A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std, ),
        ToTensorV2(),
    ]
    transf = A.Compose(
        loading_transformations + downsample_train_transformations + aug_transformations + final_transformations)
    return transf


def aggregate(x, deadzone: float, pre_mult: float, policy: str, post_mult: float, clipmargin: float, params={}):
    x = x.copy()
    if deadzone > 0:
        x = x[(x > deadzone) | (x < -deadzone)]
        if len(x) == 0:
            x = np.asarray([0, ])
    if policy == 'mean':
        x = np.mean(x)
        x = scipy.special.expit(x * pre_mult)
        x = (x - 0.5) * post_mult + 0.5
    elif policy == 'sigmean':
        x = scipy.special.expit(x * pre_mult).mean()
        x = (x - 0.5) * post_mult + 0.5
    elif policy == 'meanp':
        pow_coeff = params.pop('p', 3)
        x = np.mean(np.sign(x) * (np.abs(x) ** pow_coeff))
        x = np.sign(x) * (np.abs(x) ** (1 / pow_coeff))
        x = scipy.special.expit(x * pre_mult)
        x = (x - 0.5) * post_mult + 0.5
    elif policy == 'median':
        x = scipy.special.expit(np.median(x) * pre_mult)
        x = (x - 0.5) * post_mult + 0.5
    elif policy == 'sigmedian':
        x = np.median(scipy.special.expit(x * pre_mult))
        x = (x - 0.5) * post_mult + 0.5
    elif policy == 'maxabs':
        x = np.min(x) if abs(np.min(x)) > abs(np.max(x)) else np.max(x)
        x = scipy.special.expit(x * pre_mult)
        x = (x - 0.5) * post_mult + 0.5
    elif policy == 'avgvoting':
        x = np.mean(np.sign(x))
        x = (x * post_mult + 1) / 2
    elif policy == 'voting':
        x = np.sign(np.mean(x * pre_mult))
        x = (x - 0.5) * post_mult + 0.5
    else:
        raise NotImplementedError()
    return np.clip(x, clipmargin, 1 - clipmargin)

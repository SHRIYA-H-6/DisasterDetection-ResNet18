"""PyTorch dataset over cached pre/post pairs, with train-only augmentation."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from . import config as C
from .preprocess import cache_path

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(rgb):
    x = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def jitter(rgb, rng):
    """Random brightness/contrast jitter (independent per acquisition)."""
    alpha = rng.uniform(0.8, 1.2)  # contrast
    beta = rng.uniform(-20, 20)    # brightness
    return np.clip(rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def augment(pre, post, seg, rng):
    """Same geometric transform for pre, post and mask; photometric jitter per image."""
    if rng.random() < 0.5:
        pre, post, seg = pre[:, ::-1], post[:, ::-1], seg[:, ::-1]
    if rng.random() < 0.5:
        pre, post, seg = pre[::-1], post[::-1], seg[::-1]
    k = rng.integers(4)
    pre, post, seg = np.rot90(pre, k), np.rot90(post, k), np.rot90(seg, k)
    angle = rng.uniform(-15, 15)
    if abs(angle) > 1:
        h, w = seg.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        warp = lambda a, interp: cv2.warpAffine(np.ascontiguousarray(a), m, (w, h), flags=interp,
                                                borderMode=cv2.BORDER_REFLECT_101)
        pre, post = warp(pre, cv2.INTER_LINEAR), warp(post, cv2.INTER_LINEAR)
        seg = warp(seg, cv2.INTER_NEAREST)
    return jitter(pre, rng), jitter(post, rng), np.ascontiguousarray(seg)


class PairDataset(Dataset):
    def __init__(self, df, train=False, cache_dir=C.CACHE_DIR, seed=C.SEED):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.cache_dir = cache_dir
        self.rng = np.random.default_rng(seed)
        self.type_idx = self.df.disaster_type.map(C.DISASTER_TYPES.index).to_numpy()
        self.sev_idx = self.df.severity.map(C.SEVERITIES.index).to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        data = np.load(cache_path(self.df.sample_id[i], self.cache_dir))
        pre, post, seg = data["pre"], data["post"], data["seg"]
        if self.train:
            pre, post, seg = augment(pre, post, seg, self.rng)
        return {
            "pre": to_tensor(pre), "post": to_tensor(post),
            "seg": torch.from_numpy(seg.astype(np.int64)),
            "type": torch.tensor(self.type_idx[i]), "severity": torch.tensor(self.sev_idx[i]),
            "index": i,
        }


def balanced_sampler(df, seed=C.SEED):
    """Sample training tiles with probability inversely proportional to their combined-class size."""
    counts = df.combined_class.value_counts()
    weights = df.combined_class.map(lambda c: 1.0 / counts[c]).to_numpy(dtype=float, copy=True)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(df),
                                 replacement=True, generator=torch.Generator().manual_seed(seed))

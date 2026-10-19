"""Self-contained Dycheck-protocol NVS metrics.

Vendored verbatim (with PCK / training-only bits dropped) from
shape-of-motion's `flow3d/metrics.py` so this benchmark folder can run with
nothing more than `torch + torchmetrics + torchvision` installed (i.e. the
default `holo4d` env). No imports from `flow3d` or any other shape-of-motion
module — that repo doesn't ship as a package.

Public API:
    mPSNR(), mSSIM(), mLPIPS(net_type="alex"|"vgg"|"squeeze")

Each metric class follows torchmetrics conventions:
    metric.update(preds, targets, masks=None)   # (B, H, W, 3) float in [0, 1]
    metric.compute()                            # scalar tensor

Pass `masks=None` (or all-ones) for the "standard" un-masked PSNR/SSIM/LPIPS.
Pass `masks = valid * covisible` for the masked Dycheck variants
(mPSNR / mSSIM / mLPIPS).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image.lpips import _NoTrainLpips
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities import dim_zero_cat
from torchmetrics.utilities.imports import _TORCHVISION_AVAILABLE


class mPSNR(PeakSignalNoiseRatio):
    sum_squared_error: list[torch.Tensor]
    total: list[torch.Tensor]

    def __init__(self, **kwargs) -> None:
        super().__init__(
            data_range=1.0,
            base=10.0,
            dim=None,
            reduction="elementwise_mean",
            **kwargs,
        )
        self.add_state("sum_squared_error", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=[], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.total)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Args: preds/targets (..., 3) in [0,1]; masks (...,) binary or None."""
        if masks is None:
            masks = torch.ones_like(preds[..., 0])
        self.sum_squared_error.append(
            torch.sum(torch.pow((preds - targets) * masks[..., None], 2))
        )
        self.total.append(masks.sum().to(torch.int64) * 3)

    def compute(self) -> torch.Tensor:
        sum_squared_error = dim_zero_cat(self.sum_squared_error)
        total = dim_zero_cat(self.total)
        return -10.0 * torch.log(sum_squared_error / total).mean() / np.log(10.0)


class mSSIM(StructuralSimilarityIndexMeasure):
    similarity: list

    def __init__(self, **kwargs) -> None:
        super().__init__(
            reduction=None,
            data_range=1.0,
            return_full_image=False,
            **kwargs,
        )
        assert isinstance(self.sigma, float)

    def __len__(self) -> int:
        return sum([s.shape[0] for s in self.similarity])

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        """Args: preds/targets (B,H,W,3) in [0,1]; masks (B,H,W) or None."""
        if masks is None:
            masks = torch.ones_like(preds[..., 0])

        assert isinstance(self.kernel_size, int)
        hw = self.kernel_size // 2
        shift = (2 * hw - self.kernel_size + 1) / 2
        assert isinstance(self.sigma, float)
        f_i = (
            (torch.arange(self.kernel_size, device=preds.device) - hw + shift)
            / self.sigma
        ) ** 2
        filt = torch.exp(-0.5 * f_i)
        filt /= torch.sum(filt)

        def convolve2d(z, m, f):
            z = z.permute(0, 3, 1, 2)
            m = m[:, None]
            f = f[None, None].expand(z.shape[1], -1, -1, -1)
            z_ = F.conv2d(z * m, f, padding="valid", groups=z.shape[1])
            m_ = F.conv2d(m, torch.ones_like(f[:1]), padding="valid")
            return torch.where(
                m_ != 0, z_ * torch.ones_like(f).sum() / (m_ * z.shape[1]), 0
            ).permute(0, 2, 3, 1), (m_ != 0)[:, 0].to(z.dtype)

        filt_fn1 = lambda z, m: convolve2d(z, m, filt[:, None])
        filt_fn2 = lambda z, m: convolve2d(z, m, filt[None, :])
        filt_fn = lambda z, m: filt_fn1(*filt_fn2(z, m))

        mu0 = filt_fn(preds, masks)[0]
        mu1 = filt_fn(targets, masks)[0]
        mu00 = mu0 * mu0
        mu11 = mu1 * mu1
        mu01 = mu0 * mu1
        sigma00 = filt_fn(preds**2, masks)[0] - mu00
        sigma11 = filt_fn(targets**2, masks)[0] - mu11
        sigma01 = filt_fn(preds * targets, masks)[0] - mu01

        sigma00 = sigma00.clamp(min=0.0)
        sigma11 = sigma11.clamp(min=0.0)
        sigma01 = torch.sign(sigma01) * torch.minimum(
            torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
        )

        assert isinstance(self.data_range, float)
        c1 = (self.k1 * self.data_range) ** 2
        c2 = (self.k2 * self.data_range) ** 2
        numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
        denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
        ssim_map = numer / denom

        self.similarity.append(ssim_map.mean(dim=(1, 2, 3)))

    def compute(self) -> torch.Tensor:
        return torch.cat(self.similarity).mean()


class mLPIPS(Metric):
    sum_scores: list[torch.Tensor]
    total: list[torch.Tensor]

    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(
                "LPIPS requires torchvision (`pip install torchmetrics[image]` "
                "or `pip install torchvision`)."
            )

        valid_net_type = ("vgg", "alex", "squeeze")
        if net_type not in valid_net_type:
            raise ValueError(
                f"net_type must be one of {valid_net_type}, got {net_type}."
            )
        self.net = _NoTrainLpips(net=net_type, spatial=True)

        self.add_state("sum_scores", [], dist_reduce_fx="cat")
        self.add_state("total", [], dist_reduce_fx="cat")

    def __len__(self) -> int:
        return len(self.total)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor | None = None,
    ):
        if masks is None:
            masks = torch.ones_like(preds[..., 0])
        scores = self.net(
            (preds * masks[..., None]).permute(0, 3, 1, 2),
            (targets * masks[..., None]).permute(0, 3, 1, 2),
            normalize=True,
        )
        self.sum_scores.append((scores * masks[:, None]).sum())
        self.total.append(masks.sum().to(torch.int64))

    def compute(self) -> torch.Tensor:
        return (
            torch.tensor(self.sum_scores, device=self.device)
            / torch.tensor(self.total, device=self.device)
        ).mean()

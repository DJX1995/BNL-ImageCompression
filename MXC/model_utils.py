import argparse
import math
import random
import sys
import os
import time
import logging
import shutil
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml


from compressai.models import CompressionModel
from compressai.models.utils import conv, deconv
from compressai.layers import GDN

class Network(CompressionModel):
    def __init__(self, N=128):
        super().__init__(entropy_bottleneck_channels=N)
        self.encode = nn.Sequential(
            conv(1, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
        )

        self.decode = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 1),
        )
        self.N = N

    def forward(self, x):
       y = self.encode(x)
       y_hat, y_likelihoods = self.entropy_bottleneck(y)
       x_hat = self.decode(y_hat)
       return {
                'likelihoods':y_likelihoods,
                'x_hat':x_hat,
                }



class MainLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""
    def __init__(self, lmbda=1e-2, lmbda_fg=150, lmbda_bg_over=0.1, lmbda_fg_under=0.1, tau_fg=0.):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.lmbda_fg = lmbda_fg
        self.lmbda_bg_over = lmbda_bg_over
        self.lmbda_fg_under = lmbda_fg_under
        self.tau_fg = 0.
        self.tau_bg = 0.

    def forward(self, output, target, mask, mask_fg, psnr=False):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        out["mse_loss"] = self.mse(output["x_hat"], target)
        # make mask broadcastable
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [N,1,H,W]
        mask = mask.to(target.dtype)
        x_hat = output["x_hat"]              # [N,1,H,W]
        err   = x_hat - target               # [N,1,H,W]
        se    = err.pow(2)

        eps = 0.
        fg_mask = (mask >= 1.0 - eps).to(target.dtype)           # mask==1 → FG core
        bg_mask = (mask <= eps).to(target.dtype)                 # mask==0 → BG
        bd_mask = ((mask > eps) & (mask < 1.0 - eps)).to(target.dtype)  # boundary annulus

        # region sizes
        n_fg = fg_mask.sum().clamp_min(1e-9)
        n_bg = bg_mask.sum().clamp_min(1e-9)
        n_bd = bd_mask.sum().clamp_min(1e-9)

        # (1) Boundary: weighted symmetric MSE, boundary only
        w_bd = (1.0 + (self.lmbda_fg - 1.0) * mask) # * bd_mask
        mse_boundary = ((w_bd * se).sum() / w_bd.sum().clamp_min(1e-9)) if n_bd > 0 else \
                    torch.zeros((), device=target.device, dtype=target.dtype)
        # (2) FG core: one-sided UNDER-shoot (only if x_hat < x - tau_fg)
        undershoot = (target - x_hat - self.tau_fg).clamp_min(0.0)
        loss_fg_under = ((undershoot.pow(2) * fg_mask).sum() / n_fg) if n_fg > 0 else \
                    torch.zeros((), device=target.device, dtype=target.dtype)
        # (3) BG: one-sided OVER-shoot (only if x_hat > x + tau_bg)
        overshoot = (x_hat - target - self.tau_bg).clamp_min(0.0)
        loss_bg_over = ((overshoot.pow(2) * bg_mask).sum() / n_bg) if n_bg > 0 else \
                   torch.zeros((), device=target.device, dtype=target.dtype)
        # combine reconstruction terms
        mse_weighted = mse_boundary + self.lmbda_fg_under * loss_fg_under + self.lmbda_bg_over * loss_bg_over

        out["mse_loss"] = mse_weighted # mse_boundary

        out["loss"] = self.lmbda * (255.0 ** 2) * out["mse_loss"] + out["bpp_loss"]

        return out


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "-T",
        "--TEST",
        action='store_true',
        help='Testing'
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    given_configs, remaining = parser.parse_known_args(argv)
    with open(given_configs.config) as file:
        yaml_data= yaml.safe_load(file)
        parser.set_defaults(**yaml_data)
    args = parser.parse_args(remaining)
    return args



class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)

    def forward(self, output, target, psnr=False):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'{args.root}/{args.exp_name}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)

    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )

    return optimizer, aux_optimizer


def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    torch.save(state, base_dir+filename)
    if is_best:
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")


def load_spot_finder(file_path):
    pass
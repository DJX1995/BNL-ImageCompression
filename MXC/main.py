import tqdm
import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
from dataset import H5Dataset
import h5py
import hdf5plugin
import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2
import wandb
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader

from compressai.zoo import image_models

from model_utils import (
    AverageMeter,
    setup_logger,
    MainLoss
)
from generate_h5file import stitch_gray_grid_with_edges
from post_processing import false_positive_peak_value, plot_equal_width_hist, patch_recon_and_save_to_image
from config.base_config import config

from pathlib import Path
import sys
try:
    from BraggSpotFinder.code.eval_img_compress import SpotFinder  # absolute package import
except ModuleNotFoundError:
    # Compute repo root: this file lives at ImageCompression/Transformer_VariableROI/main.py
    REPO_ROOT = Path(__file__).resolve().parents[2]  # go up two levels to the parent that contains BraggSpotFinder/
    # Make the repo root importable (so BraggSpotFinder.* works)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "BraggSpotFinder" / "code"))
    from BraggSpotFinder.code.eval_img_compress import SpotFinder


random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    params_dict = dict(model.named_parameters())

    param_names = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_param_names = {
        n
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    params  = [params_dict[n] for n in sorted(param_names)]
    aux_params  = [params_dict[n] for n in sorted(aux_param_names)]

    loss_meter = AverageMeter()
    tqdm_emu = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Training at epoch {epoch+1}", leave=True)
    for i, batch in tqdm_emu:
        try:
            d, mask, mask_fg = batch
            d = d.to(device)
            mask = mask.to(device)
            mask_fg = mask_fg.to(device)
            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            # Check for corrupted GDN parameters before forward pass
            for name, module in model.named_modules():
                if hasattr(module, 'gamma') and hasattr(module, 'gamma_reparam'):
                    if torch.isnan(module.gamma).any() or torch.isinf(module.gamma).any():
                        print(f"WARNING: Corrupted gamma in {name}, resetting...")
                        with torch.no_grad():
                            module.gamma.data = torch.ones_like(module.gamma) * 0.1

            out_net = model(d)

            out_criterion = criterion(out_net, d, mask, mask_fg)
            loss_meter.update(out_criterion["loss"].detach().item())
            out_criterion["loss"].backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, clip_max_norm)
            optimizer.step()
            if i % 6 == 0:
                with torch.amp.autocast('cuda', enabled=False):
                    aux_loss = model.aux_loss()
                last_aux_loss = aux_loss.detach()
                aux_loss.backward()
                if clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(aux_params, clip_max_norm)
                aux_optimizer.step()
            else:
                aux_loss = last_aux_loss

            wandb.log({
                "train/loss": out_criterion["loss"].item(),
                "train/mse":  out_criterion["mse_loss"].item(),
                "train/bpp":  out_criterion["bpp_loss"].item(),
                "train/aux":  float(aux_loss.item()),
                "lr": optimizer.param_groups[0]["lr"],
                })

            update_txt=f'[{i*len(d)}/{len(train_dataloader.dataset)}] | Loss: {out_criterion["loss"].item():.3f} | MSE loss: {out_criterion["mse_loss"].item():.5f} | Bpp loss: {out_criterion["bpp_loss"].item():.4f} | Aux loss: {aux_loss.item():.2f}'

        except Exception as e:
            # surface unexpected issues (you can re-raise after printing)
            print(f"[{i}] Exception: {repr(e)}")
            raise
    logging.info(update_txt)
    return loss_meter.avg

def test_epoch(epoch, test_dataloader, model, criterion, stage='val', tqdm_meter=None):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    
    with torch.no_grad():
        tqdm_emu = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f"Evaluating at epoch {epoch+1}", leave=True)
        for i, batch in tqdm_emu:
            d, mask, mask_fg = batch
            d = d.to(device)
            mask = mask.to(device)
            mask_fg = mask_fg.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d, mask, mask_fg)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    txt = f"Eval on epoch {epoch} | Loss: {loss.avg:.3f} | MSE loss: {mse_loss.avg:.5f} | Bpp loss: {bpp_loss.avg:.4f} | Aux loss: {aux_loss.avg:.2f}"

    logging.info(txt)
    wandb.log({
        "val/loss": loss.avg,
        "val/mse":  mse_loss.avg,
        "val/bpp":  bpp_loss.avg,
        "val/aux":  aux_loss.avg,
        "epoch":    epoch,
    })
    return loss.avg


def compression_eval(test_dataloader, model, stage='val', tqdm_meter=None):
    model.eval()
    model.update()
    device = next(model.parameters()).device

    bits_compressed_full = AverageMeter()
    bits_raw_full = AverageMeter()
    bpp_compressed_full = AverageMeter()
    bpp_raw_full = AverageMeter()
    
    with torch.no_grad():
        tqdm_emu = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f"Evaluating Compression Ratio on the compression data", leave=True)
        for i, batch in tqdm_emu:
            d, _, _ = batch
            B, D, H, W = d.size()
            d = d.to(device)
            out = model.compress(d) 
            y_strings, z_strings = out["strings"]   # typically lists (batch) of bytes / nested lists

            bits_compressed = count_bits([y_strings, z_strings])
            bpp_compressed = bits_compressed / (H * W * B)
            bits_compressed_full.update(bits_compressed)
            bpp_compressed_full.update(bpp_compressed)

            bitdepth = 16  # int16 source
            bits_raw =  B * H * W * D * bitdepth
            bpp_raw  = bits_raw / (B * H * W)   # simplifies to bitdepth * channels
            bits_raw_full.update(bits_raw)
            bpp_raw_full.update(bpp_raw)

    ratio = bpp_raw_full.avg / bpp_compressed_full.avg
    txt = f"  | Compressed input bits: {bits_compressed_full.sum}, bpp: {bpp_compressed_full.avg:.3f} \
        | Raw input bits: {bits_raw_full.sum}, bpp: {bpp_raw_full.avg:.3f} \
            | Compression ratio: {ratio:.3f}"
    print(txt)
    logging.info(txt)
    wandb.log({
        "compression/bpp_compressed": bpp_compressed_full.avg,
        "compression/bpp_raw":        bpp_raw_full.avg,
        "compression/ratio":          ratio,
        })
    return ratio


def main(config):
    model_save_name = f'{config.save_name}_lambda{config.lmbda}'
    checkpoint_file = f'../experiments/models/{model_save_name}_checkpoint.pth.tar'

    logging.info('=' * 10)
    train_dataset = H5Dataset(config.dataset_path, split="train", transform=None, S=490., config=config, use_spots_region=True)
    test_dataset = H5Dataset(config.dataset_path, split='test', transform=None, S=train_dataset.S, config=config, use_spots_region=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    wandb.config.update({
    "train_size": len(train_dataloader.dataset),
    "test_size": len(test_dataloader.dataset),
    })

    #----------------build model------------------------
    net = image_models[config.model_name](quality=int(config.model_quality))
    N = net.N
    # ScaleHyperprior
    net.g_a[0] = nn.Conv2d(config.dim_in, N, kernel_size=5, stride=2, padding=5 // 2)
    net.g_s[-1] = nn.ConvTranspose2d(N, 1, kernel_size=5, stride=2, output_padding=1, padding=5 // 2)
    net = net.to(device)

    if config.resume:  # load from previous checkpoint
        logging.info('----------------Loading Model----------------')
        logging.info("Loading "+str(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, map_location=device)
        new_state_dict = checkpoint['state_dict']
        net.load_state_dict(new_state_dict)
        # Update the entropy bottleneck quantiles after loading checkpoint
        net.update()
    wandb.watch(net, log="all", log_freq=300)
    criterion = MainLoss(config.lmbda, config.lmbda_fg, config.lmbda_bg_over, config.lmbda_fg_under, config.tau_fg)

    #----------------optimizers------------------------
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
        lr=config.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=config.aux_learning_rate,
    )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.5)

    best_loss = float("inf")
    logging.info('----------------Starting training----------------')
    #----------------training------------------------
    for epoch in range(config.num_epoch):
        net.train()
        train_loss = train_one_epoch(net, criterion, train_dataloader, optimizer, aux_optimizer, epoch, config.clip_max_norm)
        lr_scheduler.step()
        if epoch >= 1:
            test_loss = test_epoch(epoch, test_dataloader, net, criterion)
            wandb.log({
                "train/epoch_loss": train_loss,
                "val/epoch_loss":   test_loss,
                "epoch": epoch,
                })
           
            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
            if config.save and is_best:
                logging.info(f'----------------finding best model at epoch {epoch}----------------')
                state_dict = net.state_dict()
                state = {
                        "epoch": epoch,
                        "state_dict": state_dict,
                        "loss": test_loss,
                        "optimizer": optimizer.state_dict(),
                        'aux_optimizer': aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    }
                torch.save(state, checkpoint_file)

                artifact = wandb.Artifact(
                    name=model_save_name,   # unique model name
                    type="model",
                    metadata={"epoch": epoch, "test_loss": float(test_loss)}
                )
                artifact.add_file(checkpoint_file)      # attach the file
                wandb.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])            # upload to W&B
                wandb.run.summary["best_test_loss"] = float(best_loss)
        else:
            wandb.log({
                "train/epoch_loss": train_loss,
                "epoch": epoch,
                })
    logging.info('----------------Compression Eval----------------')
    print('done training')
    if best_loss is None:
        # No eval ever ran; use last weights (or load a warm-start ckpt if desired)
        pass
    else:
        ckpt = torch.load(checkpoint_file, map_location=device)
        net.load_state_dict(ckpt["state_dict"])
    print('start compressing')    
    ratio = compression_eval(test_dataloader, net)
    print(f'model compression ratio {ratio:.2f}') 

# --- helpers ---
def count_bits(obj):
    """Recursively count bits in nested lists/tuples of bytes."""
    total = 0
    if isinstance(obj, (bytes, bytearray)):
        return len(obj) * 8
    if isinstance(obj, (list, tuple)):
        for x in obj:
            total += count_bits(x)
        return total
    raise TypeError(f"Unexpected type: {type(obj)}")


def eval(config, checkpoint_file=None):
    if checkpoint_file is None:
        # checkpoint_file = f'../experiments/models/gaus_plat_adaptive_hyperprior_fgrestrict_lambda10_checkpoint.pth.tar'
        checkpoint_file = f'../experiments/models/best_checkpoint.pth.tar'
    save_path = f'../experiments/deeplearning/cbass_patch_recon.h5'
    logging.info('=' * 10)
    #----------------dataset------------------------
    test_dataset = H5Dataset(config.dataset_path, split='test', transform=None, S=490, config=config, use_spots_region=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    #----------------build model------------------------
    model = image_models[config.model_name](quality=int(config.model_quality))
    N = model.N
    model.g_a[0] = nn.Conv2d(config.dim_in, N, kernel_size=5, stride=2, padding=5 // 2)
    model.g_s[-1] = nn.ConvTranspose2d(N, 1, kernel_size=5, stride=2, output_padding=1, padding=5 // 2)
    model = model.to(device)
    logging.info('----------------Loading Model----------------')
    logging.info("Loading "+str(checkpoint_file))
    checkpoint = torch.load(checkpoint_file, map_location=device)
    new_state_dict = checkpoint['state_dict']
    model.load_state_dict(new_state_dict)
    spotFinder = SpotFinder(checkpoints='../../BraggSpotFinder/best_models/model_checkpoints.pth.tar', device=device)
    logging.info('----------------Compress and Reconstruct----------------')
    model.eval()
    model.update()
    # Debug: Check model state
    logging.info("=== Model Debug Info ===")
    logging.info(f"Model device: {next(model.parameters()).device}")
    logging.info(f"Model training mode: {model.training}")

    tp_nums = AverageMeter()
    fp_nums = AverageMeter()
    fn_nums = AverageMeter()
    pred_nums = AverageMeter()
    gt_nums = AverageMeter()
    fp_max_values_list = []
    fn_max_values_list = []
    
    with torch.no_grad(), h5py.File(save_path, "w") as f:
        dset = None
        write_idx = 0
        tqdm_emu = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f"Evaluating Spot Detection on the compression data", leave=True)
        for i, batch in tqdm_emu:
            d, _, _ = batch
            B, D, H, W = d.size()
            d = d.to(device)
            # make sure decode runs in full precision
            with torch.amp.autocast('cuda', enabled=False):    # or omit autocast entirely        
                out = model.compress(d) 
                out_d = model.decompress(out["strings"], out["shape"])
                x_hat_true = out_d["x_hat"]
            
            # Fix: Handle NaN values in decompressed output
            if torch.isnan(x_hat_true).any():
                logging.info(f"Fixing NaN values in batch {i} by replacing with zeros")
                x_hat_true = torch.nan_to_num(x_hat_true, nan=0.0, posinf=1.0, neginf=0.0)
            
            # save to .h5 file
            x_hat_clamped = x_hat_true.clamp(0, 1)
            x_hat_float32 = x_hat_clamped.squeeze(1).cpu().numpy()
            # print(f"Before float16 conversion - min: {np.min(x_hat_float32):.6f}, max: {np.max(x_hat_float32):.6f}, dtype: {x_hat_float32.dtype}")
            x_hat_float32 = test_dataset.softclip_inverse(x_hat_float32)
            x_hat_float32 = np.clip(x_hat_float32, 0, np.iinfo(np.int16).max)  # avoid overflow
            x_np = np.rint(x_hat_float32).astype(np.int16)                     # round then cast
            # save to .h5 file
            
            # convert the original image to raw value range
            d_float32 = d.squeeze(1).cpu().numpy()            
            d_float32 = test_dataset.softclip_inverse(d_float32)
            d_float32 = np.clip(d_float32, 0, np.iinfo(np.int16).max)  # avoid overflow
            tp_num, fp_num, fn_num, fp_intensities, fn_intensities, gt_num, pred_num, mse_contrast_avg, details = \
                    spotFinder.compare_two_imgs(d, x_hat_clamped.detach(), img_pred_raw=x_hat_float32, img_gt_raw=d_float32)
            tp_nums.update(tp_num)
            fp_nums.update(fp_num)
            fn_nums.update(fn_num)
            gt_nums.update(gt_num)
            pred_nums.update(pred_num)
            
            # plot histogram of fp spots
            # batch_max_values = false_positive_peak_value(fp_spots, x_hat_float32)
            fp_max_values_list.extend(fp_intensities)
            fn_max_values_list.extend(fn_intensities)
            
            B, H, W = x_np.shape
            # Lazily create a resizable dataset after seeing the first batch
            if dset is None:
                # maxshape with None on axis 0 lets us append indefinitely
                # gzip + shuffle gives solid size savings with little CPU overhead
                dset = f.create_dataset(
                    "recon",
                    shape=(0, H, W),
                    maxshape=(None, H, W),
                    dtype="int16",
                    chunks=(max(1, min(8, B)), H, W),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )

            # Append this batch
            dset.resize(write_idx + B, axis=0)
            dset[write_idx : write_idx + B] = x_np
            write_idx += B
        
        ratio = compression_eval(test_dataloader, model)
        logging.info(f'model compression ratio {ratio:.2f}') 
        num_gts = gt_nums.sum
        num_pred = pred_nums.sum
        num_true_positives = tp_nums.sum
        num_false_positives = fp_nums.sum
        num_false_negatives = fn_nums.sum
        precision = num_true_positives / (num_true_positives + num_false_positives + 1e-8)
        recall = num_true_positives / (num_true_positives + num_false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        txt = f'{num_gts} spots in original image, {num_pred} spots in reconstructed images'
        logging.info(txt)
        txt = f'{num_true_positives} tps, {num_false_positives} fps, {num_false_negatives} fns'
        logging.info(txt)
        txt = f'{precision}, {recall}, {f1}'
        logging.info(txt)
    wandb.log({
        "spotfinder/precision": precision,
        "spotfinder/recall":    recall,
        "spotfinder/f1 score":  f1,
        "spotfinder/spot num in org img":  num_gts,
        "spotfinder/spot num in compress img":  num_pred,
        })
    if len(fp_max_values_list) == 0 or len(fn_max_values_list) == 0:
        logging.info("No false positives or false negatives; skipping histogram/artifact.")
        return

if __name__ == "__main__":
    setup_logger(config.log_path + time.strftime('%Y%m%d_%H%M%S') + '.log')
    use_wandb = False
    if not use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    with wandb.init(project='image-compression', name=f'{config.save_name}_lambda{config.lmbda}') as run:
        wandb.config.update(dict(config)) 
        # main(config)
        eval(config)

    # main(config)
    # eval(config)

    # # convert patch recon to image recon in .h5 file
    # test_patch_h5 = f'../data/CBASS/split2/test_patch.h5'
    # recon_patch_h5 = f'../experiments/data_recon/patch_recon.h5'
    # out_h5 = f'../experiments/deeplearning/image_recon.h5'
    # patch_recon_and_save_to_image(test_patch_h5, recon_patch_h5, out_h5)

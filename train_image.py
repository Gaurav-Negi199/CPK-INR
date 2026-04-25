#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import math
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

import torch
import torch.nn as nn

import tinycudann as tcnn
from torchinfo import summary
from torchmetrics.functional.image import (
    peak_signal_noise_ratio as psnr_torch,
    structural_similarity_index_measure as ssim_torch,
)

SCRIPTS_DIR = "./scripts"
sys.path.append(SCRIPTS_DIR)

from common import ROOT_DIR
from modules import utils


# ============================================================
# ARGUMENTS
# ============================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="tiny-cuda-nn image fitting with CNN+MLP fusion + INCODE-style adaptive sine MLP"
    )

    parser.add_argument("--input", type=str, default="0010.png")
    parser.add_argument("--epochs", type=int, default=501)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--recon_interval", type=int, default=250)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize_factor", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="Img_Rep_Output")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--coef_l2_weight", type=float, default=1e-4)
    parser.add_argument("--scheduler_eta_min_ratio", type=float, default=0.1)

    # Adaptive sine MLP settings
    parser.add_argument("--mlp_omega_0", type=float, default=30.0)
    parser.add_argument("--mlp_s0", type=float, default=0.5)

    # Coefficient regularization weights
    parser.add_argument("--a_coef", type=float, default=0.1993)
    parser.add_argument("--b_coef", type=float, default=0.0196)
    parser.add_argument("--c_coef", type=float, default=0.0588)
    parser.add_argument("--d_coef", type=float, default=0.0269)

    # Resume support
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default="",
        help="tranining_output/20260324_175750/checkpoints/checkpoint_epoch_0200.pt"
    )
    parser.add_argument(
        "--resume_run_dir",
        type=str,
        default="",
        help="Existing run directory to continue logging/saving into"
    )

    return parser.parse_args()


# ============================================================
# CONFIG
# ============================================================

config = {
    "encoding": {
        "otype": "HashGrid",
        "n_levels": 24,
        "n_features_per_level": 2,
        "log2_hashmap_size": 14,
        "base_resolution": 16,
        "per_level_scale": 1.5,
    },
}


# ============================================================
# UTILITIES
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_file):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def linear_to_srgb(img):
    limit = 0.0031308
    return np.where(
        img > limit,
        1.055 * (img ** (1.0 / 2.4)) - 0.055,
        12.92 * img
    )


def write_imagergb(file, img):
    """
    img expected in [0,1], float32
    """
    img = np.clip(img, 0.0, 1.0)
    # img = linear_to_srgb(img)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    imageio.imwrite(file, img)
    return img


def save_training_plots(metrics_csv_path, plots_dir):
    ensure_dir(plots_dir)

    epochs = []
    train_loss = []
    train_psnr = []
    train_ssim = []
    recon_psnr = []
    recon_ssim = []

    with open(metrics_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_psnr.append(float(row["train_psnr"]))
            train_ssim.append(float(row["train_ssim"]))

            rp = row["recon_psnr"]
            rs = row["recon_ssim"]

            recon_psnr.append(np.nan if rp == "" else float(rp))
            recon_ssim.append(np.nan if rs == "" else float(rs))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_loss.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_psnr, label="Train PSNR")
    valid_mask = ~np.isnan(np.array(recon_psnr))
    if np.any(valid_mask):
        plt.plot(
            np.array(epochs)[valid_mask],
            np.array(recon_psnr)[valid_mask],
            label="Reconstruction PSNR"
        )
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("PSNR vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "psnr_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    valid_mask = ~np.isnan(np.array(recon_psnr))
    if np.any(valid_mask):
        plt.plot(
            np.array(epochs)[valid_mask],
            np.array(recon_psnr)[valid_mask],
            label="Reconstruction PSNR"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction PSNR")
        plt.title("Reconstruction PSNR vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "reconstruction_psnr_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_ssim, label="Train SSIM")
    valid_mask = ~np.isnan(np.array(recon_ssim))
    if np.any(valid_mask):
        plt.plot(
            np.array(epochs)[valid_mask],
            np.array(recon_ssim)[valid_mask],
            label="Reconstruction SSIM"
        )
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "ssim_curve.png"), dpi=200)
    plt.close()


def load_checkpoint(model, optimizer, scheduler, ckpt_path, device, logger):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Scheduler state restored from checkpoint.")
    else:
        logger.warning("No scheduler_state_dict found in checkpoint. Scheduler will restart from fresh state.")

    resumed_epoch = checkpoint["epoch"]
    start_epoch = resumed_epoch + 1

    logger.info(f"Successfully resumed from epoch {resumed_epoch}")
    logger.info(f"Training will continue from epoch {start_epoch}")

    return start_epoch, resumed_epoch


# ============================================================
# IMAGE LOADING
# ============================================================

def load_rgb_image(args, logger):
    im = utils.normalize(plt.imread(args.input).astype(np.float32), True)
    im = cv2.resize(
        im,
        None,
        fx=args.resize_factor,
        fy=args.resize_factor,
        interpolation=cv2.INTER_AREA
    )

    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)

    if im.shape[-1] == 4:
        im = im[..., :3]

    H, W, _ = im.shape
    logger.info(f"Loaded resized image: H={H}, W={W}, C={im.shape[-1]}")
    return im.astype(np.float32), H, W


def make_coordinate_grid(H, W, device):
    half_dx = 0.5 / W
    half_dy = 0.5 / H

    xs = torch.linspace(half_dx, 1.0 - half_dx, W, device=device)
    ys = torch.linspace(half_dy, 1.0 - half_dy, H, device=device)

    yv, xv = torch.meshgrid(ys, xs, indexing="ij")
    xy = torch.stack((xv, yv), dim=-1)
    xy = xy.unsqueeze(0)
    return xy


# ============================================================
# MODEL COMPONENTS
# ============================================================

class SineCNN(nn.Module):
    def __init__(self, in_features, out_features, bias=False, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        self.cnn = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
        )
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.cnn.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.cnn.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.cnn(x))


class AuxMLP(nn.Module):
    def __init__(self, in_channels=64, hidden_channels=[64, 32, 4], mlp_bias=0.3120, activation_layer=nn.SiLU):
        super().__init__()

        layers = []
        curr = in_channels
        for h in hidden_channels[:-1]:
            layers.append(nn.Linear(curr, h))
            layers.append(activation_layer())
            curr = h

        layers.append(nn.Linear(curr, hidden_channels[-1]))
        self.net = nn.Sequential(*layers)

        self.net.apply(lambda m: self.init_weights(m, mlp_bias))

    @staticmethod
    def init_weights(m, mlp_bias):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, mlp_bias)

    def forward(self, x):
        return self.net(x)


class CoefficientNet(nn.Module):
    def __init__(
        self,
        gt_image_chw,
        hidden_channels=[64, 32, 4],
        mlp_bias=0.3120,
        activation_layer=nn.SiLU,
    ):
        super().__init__()

        self.register_buffer("ground_truth", gt_image_chw.unsqueeze(0))

        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:5])

        for p in self.feature_extractor.parameters():
            p.requires_grad_(False)
        self.feature_extractor.eval()

        self.gap2d = nn.AdaptiveAvgPool2d((1, 1))

        self.aux_mlp = AuxMLP(
            in_channels=64,
            hidden_channels=hidden_channels,
            mlp_bias=mlp_bias,
            activation_layer=activation_layer,
        )

        self._cached_feat = None
        self._cached_device = None

    def forward(self):
        device = self.ground_truth.device

        if self._cached_feat is None or self._cached_device != device:
            with torch.no_grad():
                feat = self.feature_extractor(self.ground_truth)
                pooled = self.gap2d(feat).flatten(1)
                self._cached_feat = pooled.detach().float()
                self._cached_device = device

        coef = self.aux_mlp(self._cached_feat)
        return coef


class AdaptiveSineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0, s0=1.0):
        super().__init__()
        self.omega_0 = omega_0
        self.s0 = s0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x, a, b, c, d):
        z = self.linear(x)

        a_bounded = 0.1 + 1.9 * torch.sigmoid(a)
        b_bounded = 0.1 + 1.9 * torch.sigmoid(b)

        return (
            a_bounded
            * torch.exp(-(self.s0 * z) ** 2)
            * torch.sin(b_bounded * self.omega_0 * z + c)
            + d
        )


class AdaptiveSineMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, num_hidden_layers=2, omega_0=30.0, s0=1.0):
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(
            AdaptiveSineLayer(in_dim, hidden_dim, is_first=True, omega_0=omega_0, s0=s0)
        )

        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(
                AdaptiveSineLayer(hidden_dim, hidden_dim, is_first=False, omega_0=omega_0, s0=s0)
            )

        self.final_linear = nn.Linear(hidden_dim, out_dim)

        with torch.no_grad():
            bound = math.sqrt(6 / hidden_dim) / max(omega_0, 1e-12)
            self.final_linear.weight.uniform_(-bound, bound)
            if self.final_linear.bias is not None:
                self.final_linear.bias.zero_()

    def forward(self, x, a, b, c, d):
        for layer in self.hidden_layers:
            x = layer(x, a, b, c, d)
        x = self.final_linear(x)
        return x


class MyHashModel(nn.Module):
    def __init__(self, gt_image_chw, n_channels=3, mlp_omega_0=30.0, mlp_s0=1.0):
        super().__init__()

        self.encoder = tcnn.Encoding(
            n_input_dims=2,
            encoding_config=config["encoding"]
        )

        self.cnn_dim = 128

        self.coef_net = CoefficientNet(
            gt_image_chw=gt_image_chw,
            hidden_channels=[64, 32, 4],
            mlp_bias=0.3120,
            activation_layer=nn.SiLU,
        )

        self.cnn = nn.Sequential(
            SineCNN(self.encoder.n_output_dims, self.cnn_dim, is_first=True),
            SineCNN(self.cnn_dim, self.cnn_dim, is_first=False),
        )

        self.mlp = AdaptiveSineMLP(
            in_dim=self.encoder.n_output_dims,
            hidden_dim=128,
            out_dim=128,
            num_hidden_layers=2,
            omega_0=mlp_omega_0,
            s0=mlp_s0,
        )

        self.cnn2 = nn.Sequential(
            SineCNN(256, self.cnn_dim, is_first=True),
            SineCNN(self.cnn_dim, self.cnn_dim, is_first=False),
        )

        self.mlp2 = AdaptiveSineMLP(
            in_dim=128,
            hidden_dim=128,
            out_dim=128,
            num_hidden_layers=2,
            omega_0=mlp_omega_0,
            s0=mlp_s0,
        )

        self.cnn_last = nn.Sequential(
            SineCNN(256, n_channels, is_first=False)
        )

    def forward(self, x):
        B, H, W, _ = x.shape
        x = x.reshape(-1, 2)

        coef = self.coef_net()
        a, b, c, d = coef[0]

        x_encode = self.encoder(x)
        x_encode_cnn = x_encode.float()

        x_cnn = x_encode_cnn.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_cnn = self.cnn(x_cnn)

        x_mlp = self.mlp(x_encode.float(), a, b, c, d)
        x_mlp_ = x_mlp.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = torch.cat((x_cnn, x_mlp_), dim=1)

        x_cnn = self.cnn2(x)

        x_mlp = self.mlp2(x_mlp, a, b, c, d)
        x_mlp_ = x_mlp.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = torch.cat((x_cnn, x_mlp_), dim=1)
        x = self.cnn_last(x)

        return x, coef


# ============================================================
# EVALUATION / RECONSTRUCTION
# ============================================================

@torch.no_grad()
def reconstruct_full_image(model, xy, device):
    model.eval()
    xy_full = xy.to(device, non_blocking=True)
    pred, _ = model(xy_full)
    pred = pred.detach().float().cpu()  # (1, C, H, W)
    return pred.squeeze(0)


@torch.no_grad()
def evaluate_reconstruction(model, xy, target_chw, epoch, recon_dir, device, logger):
    pred_chw = reconstruct_full_image(
        model=model,
        xy=xy,
        device=device,
    )

    pred_chw = torch.clamp(pred_chw, -1.0, 1.0)

    pred_for_metric = ((pred_chw + 1.0) / 2.0).unsqueeze(0)
    tgt_for_metric = ((target_chw.cpu() + 1.0) / 2.0).unsqueeze(0)

    recon_psnr = psnr_torch(pred_for_metric, tgt_for_metric, data_range=1.0).item()
    recon_ssim = ssim_torch(pred_for_metric, tgt_for_metric, data_range=1.0).item()

    pred_img = pred_for_metric.squeeze(0).permute(1, 2, 0).numpy()

    recon_path = os.path.join(recon_dir, f"recon_epoch_{epoch:04d}.png")
    write_imagergb(recon_path, pred_img)

    logger.info(
        f"[Reconstruction] Epoch {epoch} | Recon PSNR: {recon_psnr:.6f} | Recon SSIM: {recon_ssim:.6f} | Saved: {recon_path}"
    )

    return recon_psnr, recon_ssim, recon_path


def save_checkpoint(model, optimizer, scheduler, epoch, ckpt_dir, logger, args=None):
    ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:04d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args) if args is not None else None,
        },
        ckpt_path,
    )
    logger.info(f"Saved checkpoint: {ckpt_path}")


# ============================================================
# TRAIN
# ============================================================

def main():
    args = get_args()
    seed_everything(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    if args.resume_run_dir:
        base_dir = args.resume_run_dir
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(args.output_dir, run_name)

    logs_dir = os.path.join(base_dir, "logs")
    metrics_dir = os.path.join(base_dir, "metrics")
    recon_dir = os.path.join(base_dir, "reconstructions")
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    plots_dir = os.path.join(base_dir, "plots")

    for d in [logs_dir, metrics_dir, recon_dir, ckpt_dir, plots_dir]:
        ensure_dir(d)

    log_file = os.path.join(logs_dir, "training.log")
    metrics_csv = os.path.join(metrics_dir, "training_metrics.csv")

    logger = setup_logger(log_file)

    logger.info("=" * 80)
    logger.info("Training started")
    logger.info(f"Run directory: {base_dir}")
    logger.info(f"Using device: {device}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    if args.resume_checkpoint:
        logger.info(f"Resume checkpoint requested: {args.resume_checkpoint}")
    logger.info("=" * 80)

    im_np, H, W = load_rgb_image(args, logger)
    target_01 = torch.from_numpy(im_np).float()
    target_m11 = target_01 * 2.0 - 1.0
    target_chw = target_m11.permute(2, 0, 1).contiguous()
    image_batch = target_chw.unsqueeze(0)

    xy = make_coordinate_grid(H, W, device=torch.device("cpu"))

    logger.info(f"Image tensor shape: {tuple(image_batch.shape)}")
    logger.info(f"Coordinate grid shape: {tuple(xy.shape)}")

    n_channels = target_chw.shape[0]
    model = MyHashModel(
        gt_image_chw=target_chw,
        n_channels=n_channels,
        mlp_omega_0=args.mlp_omega_0,
        mlp_s0=args.mlp_s0,
    ).to(device)

    if args.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            logger.info("torch.compile enabled.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    try:
        dummy_x = torch.randn(
            1,
            H,
            W,
            2,
            device=device
        )
        logger.info("\n" + str(summary(model, input_data=dummy_x, verbose=0)))
    except Exception as e:
        logger.warning(f"Could not generate model summary: {e}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        fused=(device.type == "cuda")
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * args.scheduler_eta_min_ratio
    )

    criterion = nn.MSELoss()

    start_epoch = 1
    resumed_epoch = 0

    if args.resume_checkpoint:
        start_epoch, resumed_epoch = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=args.resume_checkpoint,
            device=device,
            logger=logger,
        )

        if start_epoch > args.epochs:
            logger.info(
                f"Checkpoint is already at epoch {resumed_epoch}, which is >= target epochs {args.epochs}. Nothing to do."
            )
            return

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total trainable parameters: {total_params}")
    logger.info(f"Approx bits-per-pixel: {total_params * 16 / (H * W):.6f}")

    if not args.resume_checkpoint:
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_psnr",
                "train_ssim",
                "recon_psnr",
                "recon_ssim",
                "epoch_time_sec",
                "total_time_sec",
                "reconstruction_image"
            ])
    else:
        if not os.path.isfile(metrics_csv):
            logger.warning(
                f"Resume requested but metrics CSV not found at {metrics_csv}. Creating a new one."
            )
            with open(metrics_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "train_psnr",
                    "train_ssim",
                    "recon_psnr",
                    "recon_ssim",
                    "epoch_time_sec",
                    "total_time_sec",
                    "reconstruction_image"
                ])

    logger.info(f"Beginning training up to epoch {args.epochs}.")
    start_time = time.perf_counter()

    force_eval_epochs = {3000, 3001}

    full_xy = xy.to(device)
    full_target = image_batch.to(device)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        output, coef = model(full_xy)
        loss_recon = criterion(output, full_target)

        a_param, b_param, c_param, d_param = coef[0]

        reg_loss = (
            args.a_coef * torch.relu(-a_param) +
            args.b_coef * torch.relu(-b_param) +
            args.c_coef * torch.relu(-c_param) +
            args.d_coef * torch.relu(-d_param)
        )

        coef_l2 = args.coef_l2_weight * (
            a_param**2 + b_param**2 + c_param**2 + d_param**2
        )

        loss = loss_recon + reg_loss + coef_l2

        if torch.isnan(loss):
            logger.error(f"NaN loss encountered at epoch {epoch}. Stopping.")
            return

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            output_metric = torch.clamp((output + 1.0) / 2.0, 0.0, 1.0)
            target_metric = torch.clamp((full_target + 1.0) / 2.0, 0.0, 1.0)

            avg_loss = loss.item()
            avg_psnr = psnr_torch(output_metric, target_metric, data_range=1.0).item()
            avg_ssim = ssim_torch(output_metric, target_metric, data_range=1.0).item()

            last_a = a_param.detach().item()
            last_b = b_param.detach().item()
            last_c = c_param.detach().item()
            last_d = d_param.detach().item()

        epoch_time = time.perf_counter() - epoch_start
        total_time = time.perf_counter() - start_time

        logger.info(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"Loss: {avg_loss:.8f} | "
            f"Train PSNR: {avg_psnr:.6f} | "
            f"Train SSIM: {avg_ssim:.6f} | "
            f"Epoch time: {timedelta(seconds=int(epoch_time))} | "
            f"Total time since resume/start: {timedelta(seconds=int(total_time))}"
        )

        a_bounded = 0.1 + 1.9 / (1.0 + math.exp(-last_a))
        b_bounded = 0.1 + 1.9 / (1.0 + math.exp(-last_b))

        logger.info(
            f"Coefficients | a={last_a:.6f}, b={last_b:.6f}, c={last_c:.6f}, d={last_d:.6f} | "
            f"a_bounded={a_bounded:.6f}, b_bounded={b_bounded:.6f}"
        )

        if epoch % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, ckpt_dir, logger, args=args)

        do_recon = (
            epoch % args.recon_interval == 0
            or epoch % args.checkpoint_interval == 0
            or epoch in force_eval_epochs
            or epoch == args.epochs
        )

        recon_psnr = ""
        recon_ssim = ""
        recon_path = ""

        if do_recon:
            recon_psnr, recon_ssim, recon_path = evaluate_reconstruction(
                model=model,
                xy=xy,
                target_chw=target_chw,
                epoch=epoch,
                recon_dir=recon_dir,
                device=device,
                logger=logger,
            )

        scheduler.step()

        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                avg_loss,
                avg_psnr,
                avg_ssim,
                recon_psnr,
                recon_ssim,
                epoch_time,
                total_time,
                recon_path
            ])

    final_ckpt = os.path.join(ckpt_dir, "final_model.pt")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
        },
        final_ckpt,
    )
    logger.info(f"Saved final model: {final_ckpt}")

    save_training_plots(metrics_csv, plots_dir)
    logger.info(f"Saved plots in: {plots_dir}")

    logger.info("=" * 80)
    logger.info("Training completed successfully.")
    logger.info(f"All outputs saved in: {base_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
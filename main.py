import argparse
import os
import numpy as np
import torch
import torch.utils.data

from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils_HSI import seed_worker
from utils.scheduler import load_scheduler
from models.get_model import get_model
from loss import make_loss
from train_pipeline import train, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bi-directional Domain Adaptation (BiDA) for Cross-domain HSI Classification"
    )
    # Model and Tokenizer options
    parser.add_argument("--model", type=str, default='BiDA', help="Model architecture")
    parser.add_argument("--tokenizer_type", type=str, default='semantic',
                        choices=['semantic', 'patch', 'band_grouping', '3d_cube'],
                        help="Tokenizer module: ['semantic', 'patch', 'band_grouping', '3d_cube']")
    
    # Dataset options
    parser.add_argument('--source_name', type=str, default='Houston13',
                        help='Name of the source domain dataset (e.g. Houston13, MFF_SD, Dioni)')
    parser.add_argument('--target_name', type=str, default='Houston18',
                        help='Name of the target domain dataset (e.g. Houston18, MFF_TD1, Loukia)')
    parser.add_argument("--dataset_dir", type=str, default='./Houston/',
                        help='Directory containing the dataset .mat files')
    parser.add_argument("--device", type=str, default="0",
                        help='GPU device ordinal (e.g. 0) or negative for CPU')
    parser.add_argument("--patch_size", type=int, default=13,
                        help='Spatial patch size (paper default: 13)')
    parser.add_argument("--num_tokens", type=int, default=4,
                        help='Number of feature tokens L (paper default: 4)')
    parser.add_argument("--dim", type=int, default=64,
                        help='Embedding dimension (paper default: 64)')
    parser.add_argument("--depth", type=int, default=3,
                        help='Number of Transformer layers (paper default: 3)')
    
    # Training & Optimization hyperparameters
    parser.add_argument("--epoch", type=int, default=200, help='Total training epochs')
    parser.add_argument("--bs", type=int, default=128, help='Batch size (paper default: 128)')
    parser.add_argument("--lr", type=float, default=1e-2, help='Base learning rate (paper default: 1e-2)')
    parser.add_argument("--momentum", type=float, default=0.9, help='SGD momentum')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='Weight decay')
    parser.add_argument("--scheduler", type=str, default='none', choices=['none', 'cosine', 'step'],
                        help='Learning rate scheduler')
    parser.add_argument("--ratio", type=float, default=0.95,
                        help='Sampling ratio for source domain training/validation split')
    parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                        help='EMA teacher model variable decay rate (default: 0.999)')
    parser.add_argument("--num_workers", type=int, default=0,
                        help='DataLoader worker count (default: 0 for cross-platform stability)')
    
    # Loss formulation & regularization weights (Paper Eq. 18)
    parser.add_argument("--loss_type", type=str, default='softmax')
    parser.add_argument("--labelsmooth", type=str, default='off')
    parser.add_argument("--lambda1", type=float, default=0.1,
                        help='Weight for MMD and Bi-distillation loss: lambda1=0.1 (Paper Eq. 18)')
    parser.add_argument("--lambda2", type=float, default=1.0,
                        help='Weight for ARS intra-domain consistency loss: lambda2=1.0 (Paper Eq. 18)')
    
    # ARS Noise Perturbations (Paper Section II-C and Table IX)
    parser.add_argument("--ars_noise", type=str, default='randomrc_gauss',
                        choices=['randomrc_gauss', 'gauss', 'randomrc', 'radiation', 'all', 'none'],
                        help="Noise pipeline for ARS consistency: ['randomrc_gauss', 'gauss', 'randomrc', 'radiation', 'all', 'none']")
    
    # Misc
    parser.add_argument("--log_interval", type=int, default=10,
                        help='Epoch interval for full metrics logging')
    parser.add_argument('--seed', type=int, default=2100, help='Random seed')
    parser.add_argument('--re_ratio', type=int, default=1, help='Source duplication factor')
    opts = parser.parse_args()

    # Assign noise_type attribute
    opts.noise_type = opts.ars_noise

    # Safe device allocation
    if torch.cuda.is_available() and int(opts.device) >= 0:
        device = torch.device(f"cuda:{opts.device}")
        print(f"[*] Running on CUDA GPU device {opts.device}")
    else:
        device = torch.device("cpu")
        print("[*] Running on CPU")

    print("=" * 60)
    print(" Bi-directional Domain Adaptation (BiDA) Framework")
    print("=" * 60)
    print(f"  Model Architecture  : {opts.model}")
    print(f"  Tokenizer Strategy  : {opts.tokenizer_type}")
    print(f"  Source Domain (SD)  : {opts.source_name}")
    print(f"  Target Domain (TD)  : {opts.target_name}")
    print(f"  Dataset Folder      : {opts.dataset_dir}")
    print(f"  Patch Size          : {opts.patch_size}x{opts.patch_size}")
    print(f"  Tokens (L) / Dim    : {opts.num_tokens} tokens / {opts.dim} dim")
    print(f"  Transformer Depth   : {opts.depth} layers")
    print(f"  Batch Size / Epochs : {opts.bs} / {opts.epoch}")
    print(f"  Learning Rate / Opt : {opts.lr} / SGD(momentum={opts.momentum}, wd={opts.weight_decay})")
    print(f"  Loss Formulation    : L_total = L_cls + {opts.lambda1}*(L_MMD + L_Bi-distill) + {opts.lambda2}*L_con")
    print(f"  ARS Noise Strategy  : {opts.ars_noise}")
    print(f"  Random Seed         : {opts.seed}")
    print("=" * 60)

    seed_worker(opts.seed)

    # Load datasets
    img_src, gt_src, labels = load_mat_hsi(opts.source_name, opts.dataset_dir, norm='normband')
    img_tar, gt_tar, _ = load_mat_hsi(opts.target_name, opts.dataset_dir, norm='normband')

    num_classes = len(labels)
    num_bands = img_src.shape[-1]
    print(f"[*] Loaded SD ({img_src.shape}) and TD ({img_tar.shape}), Classes: {num_classes}, Bands: {num_bands}")

    train_gt_src, val_gt_src = sample_gt(gt_src, opts.ratio, opts.seed, mode='random')
    test_gt_tar, _ = sample_gt(gt_tar, 1, opts.seed, mode='random')

    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src

    for _ in range(opts.re_ratio - 1):
        img_src_con = np.concatenate((img_src_con, img_src))
        train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
        val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

    r = opts.patch_size // 2
    img_src_con = np.pad(img_src_con, ((r, r), (r, r), (0, 0)), mode='reflect')
    train_gt_src_con = np.pad(train_gt_src_con, ((r, r), (r, r)), mode='reflect')
    val_gt_src_con = np.pad(val_gt_src_con, ((r, r), (r, r)), mode='reflect')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), mode='reflect')
    test_gt_tar = np.pad(test_gt_tar, ((r, r), (r, r)), mode='reflect')

    train_set = HSIDataset(img_src_con, train_gt_src_con, patch_size=opts.patch_size, data_aug=True)
    val_set = HSIDataset(img_src_con, val_gt_src_con, patch_size=opts.patch_size, data_aug=False)
    test_dataset_noise = HSIDataset(img_tar, test_gt_tar, patch_size=opts.patch_size, data_aug=True)
    test_dataset = HSIDataset(img_tar, test_gt_tar, patch_size=opts.patch_size, data_aug=False)

    g = torch.Generator()
    g.manual_seed(opts.seed)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=opts.bs, generator=g, drop_last=False, shuffle=True, num_workers=opts.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=opts.bs, generator=g, drop_last=False, shuffle=False, num_workers=opts.num_workers
    )
    test_loader_noise = torch.utils.data.DataLoader(
        test_dataset_noise, batch_size=opts.bs, generator=g, drop_last=False, shuffle=True, num_workers=opts.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opts.bs, generator=g, drop_last=False, shuffle=False, num_workers=opts.num_workers
    )

    # Initialize Student and Teacher (EMA) models
    model = get_model(opts.model, opts.source_name, opts.patch_size, opts)
    model_ema = get_model(opts.model, opts.source_name, opts.patch_size, opts, ema=True)

    model = model.to(device)
    model_ema = model_ema.to(device)

    optimizer, scheduler = load_scheduler(opts.model, model, opts)
    criterion, _ = make_loss(opts, num_classes=num_classes)

    # Checkpoint and curves output directory
    model_dir = os.path.join(
        "./checkpoints",
        f"{opts.model}_{opts.tokenizer_type}",
        f"{opts.source_name}to{opts.target_name}"
    )

    try:
        train(model, model_ema, optimizer, criterion, num_classes,
              train_loader, val_loader, test_loader_noise, test_loader,
              opts, model_dir, device, scheduler)
    except KeyboardInterrupt:
        print('\n[!] Training interrupted by user.')

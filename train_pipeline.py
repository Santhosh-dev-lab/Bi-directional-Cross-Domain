import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
import scipy.io as io

from utils.utils_HSI import grouper, sliding_window, count_sliding_window, metrics
from utils.dataset import apply_ars_noise
from loss.mmd_loss import MMD_loss


def train(network, network_ema, optimizer, criterion, num_classes,
          train_loader, val_loader, test_loader_noise, test_loader,
          opts, saving_path, device, scheduler=None):

    os.makedirs(saving_path, exist_ok=True)
    global_step = 0
    best_test_acc = 0.0

    MMD_criterion = MMD_loss()

    history = {
        'epoch': [],
        'loss_total': [],
        'loss_cls': [],
        'loss_mmd': [],
        'loss_bidistill': [],
        'loss_con': [],
        'test_oa': [],
        'test_aa': [],
        'test_kappa': []
    }

    noise_type = getattr(opts, 'noise_type', getattr(opts, 'ars_noise', 'randomrc_gauss'))
    print(f"[*] Adaptability Reinforcement Strategy (ARS) active with noise: {noise_type}")
    print(f"[*] Loss weights: lambda1={opts.lambda1} (MMD + Bi-distill), lambda2={opts.lambda2} (ARS consistency)")

    for e in tqdm(range(1, opts.epoch + 1), desc=f"BiDA [{opts.tokenizer_type}] Training"):
        network.train()

        ep_total_losses = []
        ep_cls_losses = []
        ep_mmd_losses = []
        ep_distill_losses = []
        ep_con_losses = []

        for batch_idx, (data_src, data_tar) in enumerate(zip(train_loader, test_loader_noise)):
            images_src, targets_src = data_src
            images_tar, targets_tar = data_tar

            images_src = images_src.to(device)
            targets_src = targets_src.to(device)
            images_tar = images_tar.to(device)

            # Generate ARS noise perturbations o1 (teacher) and o2 (student)
            images_src_std = apply_ars_noise(images_src, noise_type=noise_type)
            images_tar_std = apply_ars_noise(images_tar, noise_type=noise_type)

            images_src_tec = apply_ars_noise(images_src, noise_type=noise_type)
            images_tar_tec = apply_ars_noise(images_tar, noise_type=noise_type)

            optimizer.zero_grad()

            # 1. Student Forward: receives noise perturbation o2
            out_x, out_x_tar, out_x_fusion, out_x_fusion_src, (feat_s, feat_t, feat_st, feat_ts) = network(
                images_src_std, images_tar_std, return_features=True
            )

            # 2. Teacher (EMA) Forward: receives noise perturbation o1
            with torch.no_grad():
                _, _, _, _, (feat_s_ema, feat_t_ema, _, _) = network_ema(
                    images_src_tec, images_tar_tec, return_features=True
                )

            # 3. Supervised Classification Loss on SD (Eq. 8)
            loss_cls = criterion(out_x, targets_src)

            # 4. Bi-directional Distillation Loss (Eq. 10, 11, 12)
            # Soft labels from coupled branch guide source and target branches
            loss_dis_t = distill_loss(out_x_fusion, out_x_tar)      # Distill(T_s->t^cls, T_t^cls)
            loss_dis_s = distill_loss(out_x_fusion_src, out_x)      # Distill(T_t->s^cls, T_s^cls)
            loss_bidistill = loss_dis_t + loss_dis_s

            # 5. Maximum Mean Discrepancy (MMD) across feature token spaces (Eq. 13, 14)
            # Marginal distribution difference in Hilbert space between SD & TD adaptive spaces
            loss_mmd_branch = MMD_criterion(feat_s, feat_t)
            loss_mmd_coupled = MMD_criterion(feat_st, feat_ts)
            loss_mmd = loss_mmd_branch + loss_mmd_coupled

            # 6. Adaptability Reinforcement Strategy (ARS) intra-domain consistency (Eq. 15, 16, 17)
            # Mean square error between teacher model (o1) and student model (o2)
            loss_con_src = F.mse_loss(feat_s, feat_s_ema)
            loss_con_tar = F.mse_loss(feat_t, feat_t_ema)
            loss_con = loss_con_src + loss_con_tar

            # 7. Total Unified Loss (Eq. 18):
            # L_total = L_cls + lambda1 * (L_MMD + L_Bi-distill) + lambda2 * L_con
            loss_total = loss_cls + opts.lambda1 * (loss_mmd + loss_bidistill) + opts.lambda2 * loss_con

            loss_total.backward()
            optimizer.step()

            global_step += 1
            update_ema_variables(network, network_ema, opts.ema_decay, global_step)

            ep_total_losses.append(loss_total.item())
            ep_cls_losses.append(loss_cls.item())
            ep_mmd_losses.append(loss_mmd.item())
            ep_distill_losses.append(loss_bidistill.item())
            ep_con_losses.append(loss_con.item())

        mean_total_loss = float(np.mean(ep_total_losses))
        mean_cls_loss = float(np.mean(ep_cls_losses))
        mean_mmd_loss = float(np.mean(ep_mmd_losses))
        mean_distill_loss = float(np.mean(ep_distill_losses))
        mean_con_loss = float(np.mean(ep_con_losses))

        if scheduler is not None:
            scheduler.step()

        # Evaluate target test accuracy per epoch
        ts_acc, results = validation(network, test_loader, device, num_classes, show=(e % opts.log_interval == 0))
        mean_aa = float(np.mean(results.get('TPR', [0]))) * 100.0 if 'TPR' in results else 0.0
        kappa = float(results.get('Kappa', 0.0))

        # Record metrics per epoch
        history['epoch'].append(e)
        history['loss_total'].append(mean_total_loss)
        history['loss_cls'].append(mean_cls_loss)
        history['loss_mmd'].append(mean_mmd_loss)
        history['loss_bidistill'].append(mean_distill_loss)
        history['loss_con'].append(mean_con_loss)
        history['test_oa'].append(ts_acc)
        history['test_aa'].append(mean_aa)
        history['test_kappa'].append(kappa)

        # Check for best target test accuracy
        is_best = ts_acc > best_test_acc
        if is_best:
            best_test_acc = ts_acc
            save_ts_checkpoint(network, is_best, saving_path,
                               epoch=e, acc=best_test_acc, tmp_acc=ts_acc, seed=opts.seed)

        # Periodic logging and curve update
        if e % opts.log_interval == 0 or e == 1 or e == opts.epoch:
            log_str = (f"Epoch {e:3d}/{opts.epoch} | Loss: {mean_total_loss:.4f} "
                       f"[Cls: {mean_cls_loss:.4f}, MMD: {mean_mmd_loss:.4f}, "
                       f"Distill: {mean_distill_loss:.4f}, Con: {mean_con_loss:.4f}] | "
                       f"Test OA: {ts_acc * 100:.2f}% (Best: {best_test_acc * 100:.2f}%) | "
                       f"Kappa: {kappa:.4f}")
            tqdm.write(log_str)

        # Save training dynamics curves and history
        plot_training_curves(history, saving_path)
        with open(os.path.join(saving_path, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    # Final summary and mat save
    io.savemat(os.path.join(saving_path, f'final_results_{opts.seed}.mat'), {
        'lr': opts.lr, 'lambda1': opts.lambda1, 'lambda2': opts.lambda2,
        'tokenizer_type': opts.tokenizer_type, 'depth': opts.depth,
        'best_test_acc': best_test_acc, 'history': history
    })
    print(f"\n[+] Training complete! Best Test OA: {best_test_acc * 100:.2f}%. Saved to {saving_path}")
    return best_test_acc, history


def validation(network, val_loader, device, num_classes, show=False):
    network.eval()
    num_correct = 0.0
    total_num = 0.0
    ps = []
    ys = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            out = network(images, images, inference_target_only=True)
            if isinstance(out, (tuple, list)):
                outputs = out[1] if out[1] is not None else out[0]
            else:
                outputs = out

            _, preds = torch.max(outputs, dim=1)
            ps.append(preds.detach().cpu().numpy())
            ys.append(targets.detach().cpu().numpy())

            num_correct += (preds == targets).sum().item()
            total_num += targets.size(0)

    overall_acc = num_correct / max(total_num, 1.0)
    ps = np.concatenate(ps) if ps else np.array([])
    ys = np.concatenate(ys) if ys else np.array([])

    results = metrics(ps, ys, n_classes=num_classes)
    if show:
        tqdm.write(f"Confusion Matrix:\n{results.get('Confusion_matrix', '')}")
        tqdm.write(f"TPR (AA): {np.round(results.get('TPR', np.array([])) * 100, 2)}")
        tqdm.write(f"OA: {results.get('Accuracy', 0):.4f} | Kappa: {results.get('Kappa', 0):.4f}")

    return overall_acc, results


def test(network, model_dir, image, patch_size, n_classes, device):
    network.load_state_dict(torch.load(os.path.join(model_dir, "model_best.pth"), map_location=device))
    network.eval()

    batch_size = 64
    window_size = (patch_size, patch_size)
    image_w, image_h = image.shape[:2]
    pad_size = patch_size // 2

    image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    probs = np.zeros(image.shape[:2] + (n_classes,))

    iterations = count_sliding_window(image, window_size=window_size) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(image, window_size=window_size)),
                      total=iterations, desc="Inference on HSI"):
        with torch.no_grad():
            data = [b[0] for b in batch]
            data = np.copy(data).transpose((0, 3, 1, 2))
            data = torch.from_numpy(data).unsqueeze(1).to(device)

            indices = [b[1:] for b in batch]
            output = network(data, data, inference_target_only=True)
            if isinstance(output, (tuple, list)):
                output = output[1] if output[1] is not None else output[0]
            output = output.to('cpu').numpy()

            for (x, y, w, h), out in zip(indices, output):
                probs[x + w // 2, y + h // 2] += out

    return probs[pad_size:image_w + pad_size, pad_size:image_h + pad_size, :]


def save_ts_checkpoint(network, is_best, saving_path, **kwargs):
    os.makedirs(saving_path, exist_ok=True)
    if is_best:
        tqdm.write(">>> New Best Test OA = {acc:.4f} at epoch {epoch}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(
            saving_path, 'model_ts_best{acc:.4f}_{seed}.pth'.format(**kwargs)))
        torch.save(network.state_dict(), os.path.join(saving_path, 'model_best.pth'))


def update_ema_variables(model, ema_model, alpha, global_step):
    """EMA for teacher model parameters"""
    alpha = min(1.0 - 1.0 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1.0 - alpha)


def distill_loss(teacher_output, student_out):
    """KL divergence with soft labels output from coupled branch"""
    teacher_prob = F.softmax(teacher_output, dim=-1)
    loss = torch.sum(-teacher_prob * F.log_softmax(student_out, dim=-1), dim=-1)
    return loss.mean()


def plot_training_curves(history, saving_path):
    epochs = history['epoch']
    if len(epochs) == 0:
        return

    os.makedirs(saving_path, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(epochs, history['loss_total'], label=r'$\mathcal{L}_{total}$', color='black', linewidth=1.8)
    axes[0].plot(epochs, history['loss_cls'], label=r'$\mathcal{L}_{cls}$', linestyle='--', color='royalblue')
    axes[0].plot(epochs, history['loss_mmd'], label=r'$\mathcal{L}_{MMD}$', linestyle='-.', color='crimson')
    axes[0].plot(epochs, history['loss_bidistill'], label=r'$\mathcal{L}_{Bi-distill}$', linestyle=':', color='darkorange')
    axes[0].plot(epochs, history['loss_con'], label=r'$\mathcal{L}_{con}$ (ARS)', linestyle='-', color='forestgreen', alpha=0.8)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('BiDA Training Loss Dynamics', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Test Accuracy curves
    oa_pct = [acc * 100 for acc in history['test_oa']]
    axes[1].plot(epochs, oa_pct, label='Overall Accuracy (OA %)', color='navy', linewidth=1.8)
    if any(history['test_aa']):
        axes[1].plot(epochs, history['test_aa'], label='Average Accuracy (AA %)', color='teal', linestyle='--')
    if any(history['test_kappa']):
        kappa_pct = [k * 100 for k in history['test_kappa']]
        axes[1].plot(epochs, kappa_pct, label='Kappa Coefficient (x100)', color='purple', linestyle='-.')

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%) / Metric', fontsize=12)
    axes[1].set_title('Target Domain Test Dynamics', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(saving_path, 'training_dynamics.png'), dpi=200)
    plt.close()

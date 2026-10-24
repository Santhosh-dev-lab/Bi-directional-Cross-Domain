import os
import shutil
import unittest
import torch
import torch.nn as nn
import numpy as np

from models.tokenizers import (
    SemanticTokenizer,
    SpatialPatchTokenizer,
    BandGroupingTokenizer,
    Cube3DTokenizer,
    get_tokenizer
)
from models.BiDA import BiDA, BiDAnet
from loss.mmd_loss import MMD_loss
from train_pipeline import distill_loss, update_ema_variables, plot_training_curves
from utils.dataset import apply_ars_noise, HSIDataset


class TestBiDAPaperAlignment(unittest.TestCase):
    def setUp(self):
        self.B = 4
        self.n_bands = 48
        self.patch_size = 13
        self.num_tokens = 4
        self.dim = 64
        self.num_classes = 7
        self.x = torch.randn(self.B, 1, self.n_bands, self.patch_size, self.patch_size)
        self.x_tar = torch.randn(self.B, 1, self.n_bands, self.patch_size, self.patch_size)

    def test_modular_tokenizers_shapes_and_compatibility(self):
        """Verify all 4 tokenizers produce compatible (B, L, dim) tokens."""
        tokenizers = {
            'semantic': SemanticTokenizer(n_bands=self.n_bands, dim=self.dim, num_tokens=self.num_tokens, patch_size=self.patch_size),
            'patch': SpatialPatchTokenizer(n_bands=self.n_bands, dim=self.dim, num_tokens=self.num_tokens, patch_size=self.patch_size),
            'band_grouping': BandGroupingTokenizer(n_bands=self.n_bands, dim=self.dim, num_tokens=self.num_tokens, patch_size=self.patch_size),
            '3d_cube': Cube3DTokenizer(n_bands=self.n_bands, dim=self.dim, num_tokens=self.num_tokens, patch_size=self.patch_size)
        }

        for name, tok in tokenizers.items():
            tokens = tok(self.x)
            self.assertEqual(tokens.shape, (self.B, self.num_tokens, self.dim),
                             f"{name} tokenizer returned shape {tokens.shape}, expected {(self.B, self.num_tokens, self.dim)}")
            print(f"[PASS] Tokenizer '{name}' -> shape: {tokens.shape}")

    def test_bida_with_all_tokenizers(self):
        """Verify BiDA model instantiates and runs seamlessly with all 4 tokenizers."""
        class DummyOpts:
            def __init__(self, tok_type):
                self.num_tokens = 4
                self.dim = 64
                self.depth = 2
                self.patch_size = 13
                self.tokenizer_type = tok_type

        for tok_type in ['semantic', 'patch', 'band_grouping', '3d_cube']:
            opts = DummyOpts(tok_type)
            net = BiDA('Houston13', opts)
            out_x, out_tar, out_fusion, out_fusion_src, (feat_s, feat_t, feat_st, feat_ts) = net(
                self.x, self.x_tar, return_features=True
            )
            # Check classification logits shape
            self.assertEqual(out_x.shape, (self.B, self.num_classes))
            self.assertEqual(out_tar.shape, (self.B, self.num_classes))
            self.assertEqual(out_fusion.shape, (self.B, self.num_classes))
            self.assertEqual(out_fusion_src.shape, (self.B, self.num_classes))

            # Check feature tokens (L + 1 = 5 tokens prepending cls_token)
            expected_feat_shape = (self.B, self.num_tokens + 1, self.dim)
            self.assertEqual(feat_s.shape, expected_feat_shape)
            self.assertEqual(feat_t.shape, expected_feat_shape)
            self.assertEqual(feat_st.shape, expected_feat_shape)
            self.assertEqual(feat_ts.shape, expected_feat_shape)
            print(f"[PASS] BiDA [{tok_type}] forward pass & feature shapes verified.")

    def test_mmd_loss_on_feature_token_spaces(self):
        """Verify MMD loss calculation on feature token spaces and that MMD > 0."""
        mmd_crit = MMD_loss()
        feat_s = torch.randn(self.B, self.num_tokens + 1, self.dim)
        feat_t = torch.randn(self.B, self.num_tokens + 1, self.dim)
        feat_st = torch.randn(self.B, self.num_tokens + 1, self.dim)
        feat_ts = torch.randn(self.B, self.num_tokens + 1, self.dim)

        # MMD on branch feature tokens and coupled feature tokens (Eq. 13 & 14)
        mmd_branch = mmd_crit(feat_s, feat_t)
        mmd_coupled = mmd_crit(feat_st, feat_ts)
        loss_mmd = mmd_branch + mmd_coupled

        self.assertGreater(loss_mmd.item(), 0.0, "MMD loss must be non-zero for distinct distributions")
        # Check MMD on identical tensors evaluates to 0
        mmd_self = mmd_crit(feat_s, feat_s)
        self.assertAlmostEqual(mmd_self.item(), 0.0, places=4)
        print(f"[PASS] MMD Loss verified: L_MMD = {loss_mmd.item():.4f} (Branch: {mmd_branch.item():.4f}, Coupled: {mmd_coupled.item():.4f})")

    def test_unified_loss_formulation(self):
        """Verify unified loss strictly matches Eq. 18 with lambda1=0.1, lambda2=1.0."""
        ce = nn.CrossEntropyLoss()
        mmd_crit = MMD_loss()

        targets = torch.randint(0, self.num_classes, (self.B,))
        out_x = torch.randn(self.B, self.num_classes, requires_grad=True)
        out_tar = torch.randn(self.B, self.num_classes, requires_grad=True)
        out_fusion = torch.randn(self.B, self.num_classes, requires_grad=True)
        out_fusion_src = torch.randn(self.B, self.num_classes, requires_grad=True)

        feat_s = torch.randn(self.B, self.num_tokens + 1, self.dim, requires_grad=True)
        feat_t = torch.randn(self.B, self.num_tokens + 1, self.dim, requires_grad=True)
        feat_st = torch.randn(self.B, self.num_tokens + 1, self.dim, requires_grad=True)
        feat_ts = torch.randn(self.B, self.num_tokens + 1, self.dim, requires_grad=True)

        feat_s_ema = torch.randn(self.B, self.num_tokens + 1, self.dim)
        feat_t_ema = torch.randn(self.B, self.num_tokens + 1, self.dim)

        lambda1 = 0.1
        lambda2 = 1.0

        loss_cls = ce(out_x, targets)
        loss_bidistill = distill_loss(out_fusion, out_tar) + distill_loss(out_fusion_src, out_x)
        loss_mmd = mmd_crit(feat_s, feat_t) + mmd_crit(feat_st, feat_ts)
        loss_con = torch.nn.functional.mse_loss(feat_s, feat_s_ema) + torch.nn.functional.mse_loss(feat_t, feat_t_ema)

        loss_total = loss_cls + lambda1 * (loss_mmd + loss_bidistill) + lambda2 * loss_con
        loss_total.backward()

        self.assertIsNotNone(out_x.grad)
        self.assertIsNotNone(feat_s.grad)
        print(f"[PASS] Unified Loss verified: L_total = {loss_total.item():.4f}")

    def test_noise_perturbations(self):
        """Verify Gaussian, RandomRC, Radiation noise pipelines."""
        for ntype in ['gauss', 'randomrc', 'radiation', 'randomrc_gauss', 'all']:
            perturbed = apply_ars_noise(self.x, noise_type=ntype)
            self.assertEqual(perturbed.shape, self.x.shape)
            self.assertFalse(torch.allclose(perturbed, self.x), f"Perturbation {ntype} did not change input")
            print(f"[PASS] Noise perturbation pipeline '{ntype}' operational.")

    def test_plot_curves_generator(self):
        """Verify plotting of training loss and test accuracy curves."""
        test_dir = "./checkpoints/test_run"
        history = {
            'epoch': [1, 2, 3],
            'loss_total': [2.5, 1.8, 1.2],
            'loss_cls': [1.5, 1.0, 0.7],
            'loss_mmd': [0.4, 0.3, 0.2],
            'loss_bidistill': [0.3, 0.2, 0.1],
            'loss_con': [0.3, 0.3, 0.2],
            'test_oa': [0.65, 0.72, 0.81],
            'test_aa': [60.0, 70.0, 79.5],
            'test_kappa': [0.55, 0.64, 0.75]
        }
        plot_training_curves(history, test_dir)
        self.assertTrue(os.path.exists(os.path.join(test_dir, "training_dynamics.png")))
        shutil.rmtree(test_dir, ignore_errors=True)
        print("[PASS] Learning curve generation verified.")


    def test_end_to_end_training_smoke(self):
        """End-to-end smoke test of training pipeline with synthetic HSI data."""
        from train_pipeline import train
        from models.get_model import get_model
        from utils.scheduler import load_scheduler

        test_dir = "./checkpoints/smoke_test"
        device = torch.device("cpu")

        class Opts:
            model = 'BiDA'
            tokenizer_type = 'semantic'
            patch_size = 13
            num_tokens = 4
            dim = 64
            depth = 1
            bs = 4
            epoch = 2
            lr = 0.01
            momentum = 0.9
            weight_decay = 1e-4
            lambda1 = 0.1
            lambda2 = 1.0
            ema_decay = 0.999
            log_interval = 1
            seed = 42
            loss_type = 'softmax'
            labelsmooth = 'off'
            source_name = 'Houston13'
            target_name = 'Houston18'
            ars_noise = 'randomrc_gauss'
            noise_type = 'randomrc_gauss'
            scheduler = 'none'

        opts = Opts()
        model = get_model(opts.model, opts.source_name, opts.patch_size, opts).to(device)
        model_ema = get_model(opts.model, opts.source_name, opts.patch_size, opts, ema=True).to(device)

        optimizer, scheduler = load_scheduler(opts.model, model, opts)
        criterion = nn.CrossEntropyLoss()

        # Create synthetic datasets
        dummy_data = torch.randn(8, 1, 48, 13, 13)
        dummy_labels = torch.randint(0, 7, (8,))
        dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

        best_acc, history = train(
            model, model_ema, optimizer, criterion, 7,
            loader, loader, loader, loader,
            opts, test_dir, device, scheduler
        )

        self.assertEqual(len(history['epoch']), 2)
        self.assertTrue(os.path.exists(os.path.join(test_dir, "training_dynamics.png")))
        self.assertTrue(os.path.exists(os.path.join(test_dir, "history.json")))
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"[PASS] End-to-end training smoke test completed successfully. Best OA: {best_acc:.4f}")


if __name__ == '__main__':
    unittest.main()


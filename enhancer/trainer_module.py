import os
import torch
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional.classification import accuracy

from .ssim import SSIM, MS_SSIM
from .config import TrainerConfig, TrainingMode
from .dataset import FrameDataset  # tylko do zapisu pełnych klatek w predict_step


class TrainerModule(pl.LightningModule):
    def __init__(
        self,
        config: TrainerConfig,
        enhancer,
        discriminator,
        test_full_frames: bool = False,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.enhancer = enhancer
        self.discriminator = discriminator
        self.mode = config.mode
        self.config = config.current

        self.separation_epochs = config.separation_epochs

        self.enhancer_lr = self.config.enhancer_lr
        self.discriminator_lr = self.config.discriminator_lr

        self.betas = self.config.betas
        self.momentum = self.config.momentum
        self.discriminator_min_loss = self.config.discriminator_min_loss
        self.enhancer_min_loss = self.config.enhancer_min_loss

        self.enhancer_losses = [1.0]
        self.discriminator_losses = [1.0]
        self.probe = self.config.probe

        self.num_samples = self.config.num_samples

        # SSIM modules - jak u Piotra (per_channel=True, channel=3)
        self.ssim = SSIM(data_range=1.0, win_size=9, channel=3, per_channel=True)
        self.msssim = MS_SSIM(data_range=1.0, win_size=9, channel=3, per_channel=True)

        self.test_full_frames = test_full_frames
        self.channels_grad_scales = config.channels_grad_scales

    def forward(self, chunks, metadata, vvc_features=None):
        return self.enhancer(chunks, metadata, vvc_features)

    def adversarial_loss(self, y_hat, y):
        # BCE z logitami – bezpieczne przy AMP
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def what_to_train(self):
        if self.mode == TrainingMode.ENHANCER:
            return True, None
        elif self.mode == TrainingMode.DISCRIMINATOR:
            return None, True
        else:
            e_loss = np.mean(self.enhancer_losses)
            d_loss = np.mean(self.discriminator_losses)

            e_train = e_loss >= self.enhancer_min_loss
            d_train = d_loss >= self.discriminator_min_loss

            if not e_train and not d_train:
                return True, True

            return bool(e_train), bool(d_train)

    def g_step(self, chunks, orig_chunks, metadata, vvc_features=None, stage="train"):
        enhanced = self(chunks, metadata, vvc_features)
        prefix = "" if stage == "train" else stage + "_"

        # label dla GAN
        valid = torch.ones(chunks.size(0), 1, device=chunks.device, dtype=chunks.dtype)

        # rozdzielenie kanałów
        eY, eU, eV = enhanced[:, [0]], enhanced[:, [1]], enhanced[:, [2]]
        oY, oU, oV = orig_chunks[:, [0]], orig_chunks[:, [1]], orig_chunks[:, [2]]

        channels_grad_scales = torch.tensor(
            self.channels_grad_scales,
            device=enhanced.device,
            dtype=enhanced.dtype,
        )

        # SSIM per-channel (3 wartości)
        ssim_l = 1 - self.ssim(orig_chunks, enhanced)  # (3,)
        ssimY, ssimU, ssimV = ssim_l
        ssim_loss = (channels_grad_scales * ssim_l).sum()

        self.log(f"{prefix}g_ssimY_loss", ssimY, prog_bar=False)
        self.log(f"{prefix}g_ssimU_loss", ssimU, prog_bar=False)
        self.log(f"{prefix}g_ssimV_loss", ssimV, prog_bar=False)
        self.log(f"{prefix}g_ssim_loss", ssim_loss, prog_bar=False)
        self.log(f"{prefix}g_ssim", ssim_l.mean(), prog_bar=False)

        # MS-SSIM
        msssim_l = 1 - self.msssim(orig_chunks, enhanced)
        msssimY, msssimU, msssimV = msssim_l
        msssim_loss = (channels_grad_scales * msssim_l).sum()

        self.log(f"{prefix}g_msssimY_loss", msssimY, prog_bar=False)
        self.log(f"{prefix}g_msssimU_loss", msssimU, prog_bar=False)
        self.log(f"{prefix}g_msssimV_loss", msssimV, prog_bar=False)
        self.log(f"{prefix}g_msssim_loss", msssim_loss, prog_bar=False)
        self.log(f"{prefix}g_msssim", msssim_l.mean(), prog_bar=False)

        # MSE per-channel
        mseY = F.mse_loss(oY, eY)
        mseU = F.mse_loss(oU, eU)
        mseV = F.mse_loss(oV, eV)
        mse_per_channel = torch.stack([mseY, mseU, mseV])
        mse_loss = (channels_grad_scales * mse_per_channel).sum()
        self.log(f"{prefix}g_mse_loss", mse_loss, prog_bar=False)
        self.log(f"{prefix}g_mseY_loss", mseY, prog_bar=False)
        self.log(f"{prefix}g_mseU_loss", mseU, prog_bar=False)
        self.log(f"{prefix}g_mseV_loss", mseV, prog_bar=False)

        # L1 per-channel
        l1Y = F.l1_loss(oY, eY)
        l1U = F.l1_loss(oU, eU)
        l1V = F.l1_loss(oV, eV)
        l1_per_channel = torch.stack([l1Y, l1U, l1V])
        l1_loss = (channels_grad_scales * l1_per_channel).sum()
        self.log(f"{prefix}g_l1_loss", l1_loss, prog_bar=False)
        self.log(f"{prefix}g_l1Y_loss", l1Y, prog_bar=False)
        self.log(f"{prefix}g_l1U_loss", l1U, prog_bar=False)
        self.log(f"{prefix}g_l1V_loss", l1V, prog_bar=False)

        # GAN vs enhancer-only
        if (
            self.mode == TrainingMode.GAN
            and self.current_epoch > self.separation_epochs
        ):
            preds = self.discriminator(enhanced)
            gd_loss = self.adversarial_loss(preds, valid)

            g_loss = (
                0.08 * msssim_loss
                + 0.08 * ssim_loss
                + 0.29 * mse_loss
                + 0.45 * l1_loss
                + 0.1 * gd_loss
            )
            self.log(f"{prefix}g_d_loss", gd_loss, prog_bar=True)
            self.enhancer_losses.append(gd_loss.item())
            self.enhancer_losses = self.enhancer_losses[: self.probe]
        else:
            preds = None
            # wariant "tylko enhancer" (bez GAN)
            g_loss = 0.5 * l1_loss + 0.3 * mse_loss + 0.2 * ssim_loss

        self.log(f"{prefix}g_loss", g_loss, prog_bar=True)

        if stage != "train":
            cY, cU, cV = chunks[:, [0]], chunks[:, [1]], chunks[:, [2]]

            enhanced_psnr = psnr(enhanced, orig_chunks)
            orig_psnr = psnr(chunks, orig_chunks)

            epsnrY = psnr(eY, oY)
            epsnrU = psnr(eU, oU)
            epsnrV = psnr(eV, oV)

            cpsnrY = psnr(cY, oY)
            cpsnrU = psnr(cU, oU)
            cpsnrV = psnr(cV, oV)

            enhanced_ssim = ssim(enhanced, orig_chunks)
            orig_ssim = ssim(chunks, orig_chunks)

            self.log_dict(
                {
                    f"{prefix}psnr": enhanced_psnr,
                    f"{prefix}psnrY": epsnrY,
                    f"{prefix}psnrU": epsnrU,
                    f"{prefix}psnrV": epsnrV,
                    f"{prefix}ref_psnr": orig_psnr,
                    f"{prefix}ref_psnrY": cpsnrY,
                    f"{prefix}ref_psnrU": cpsnrU,
                    f"{prefix}ref_psnrV": cpsnrV,
                    f"{prefix}ssim": enhanced_ssim,
                    f"{prefix}ref_ssim": orig_ssim,
                },
            )

        return enhanced, preds, g_loss

    def d_step(self, fake_chunks, orig_chunks, stage="train"):
        prefix = "" if stage == "train" else stage + "_"

        valid = torch.ones(orig_chunks.size(0), 1, device=orig_chunks.device, dtype=orig_chunks.dtype)
        real_pred = self.discriminator(orig_chunks)
        real_loss = self.adversarial_loss(real_pred, valid)
        real_accuracy = accuracy(real_pred, valid, task="binary")

        fake = torch.zeros(orig_chunks.size(0), 1, device=orig_chunks.device, dtype=orig_chunks.dtype)
        fake_preds = self.discriminator(fake_chunks)
        fake_loss = self.adversarial_loss(fake_preds, fake)
        fake_accuracy = accuracy(fake_preds, fake, task="binary")

        acc = (real_accuracy + fake_accuracy) / 2
        d_loss = (real_loss + fake_loss) / 2

        self.discriminator_losses.append(d_loss.item())
        self.discriminator_losses = self.discriminator_losses[: self.probe]

        self.log(f"{prefix}d_loss", d_loss, prog_bar=True)
        self.log(f"{prefix}d_real_loss", real_loss, prog_bar=False)
        self.log(f"{prefix}d_fake_loss", fake_loss, prog_bar=False)
        self.log(f"{prefix}d_real_acc", real_accuracy, prog_bar=False)
        self.log(f"{prefix}d_fake_acc", fake_accuracy, prog_bar=False)
        self.log(f"{prefix}d_acc", acc, prog_bar=True)

        return fake_preds, real_pred, d_loss

    def log_images(
        self, enhanced, chunks, orig_chunks, preds, real_preds, stage="train"
    ):
        # brak loggera -> nic nie robimy
        if (
            self.logger is None
            or not hasattr(self.logger, "experiment")
            or not hasattr(self.logger.experiment, "log")
        ):
            return

        prefix = "" if stage == "train" else stage + "_"

        log = {"uncompressed": [], "decompressed": []}
        if self.mode != TrainingMode.DISCRIMINATOR and enhanced is not None:
            log["enhanced"] = []

        actual_samples = min(self.num_samples, orig_chunks.size(0))

        for i in range(actual_samples):
            orig = orig_chunks[i].cpu()
            dec = chunks[i].cpu()

            if self.mode != TrainingMode.DISCRIMINATOR and enhanced is not None:
                enh = enhanced[i].cpu()
                log["enhanced"].append(
                    wandb.Image(
                        enh,
                        caption=f"Pred: {preds[i].item()}"
                        if self.mode == TrainingMode.GAN and preds is not None
                        else f"ENH: {i}",
                    )
                )

            log["uncompressed"].append(
                wandb.Image(
                    orig,
                    caption=f"Pred: {real_preds[i].item()}"
                    if self.mode != TrainingMode.ENHANCER and real_preds is not None
                    else f"UNC: {i}",
                )
            )

            log["decompressed"].append(
                wandb.Image(
                    dec,
                    caption=f"Pred: {preds[i].item()}"
                    if self.mode == TrainingMode.DISCRIMINATOR and preds is not None
                    else f"DEC: {i}",
                )
            )

        log = {prefix + key: value for key, value in log.items()}
        self.logger.experiment.log(log)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        # trening ZAWSZE na chunks_pt -> batch ma 5 elementów
        chunks, orig_chunks, metadata, _, vvc_features = batch

        e_train, d_train = self.what_to_train()
        preds = None

        # ENHANCER
        if e_train:
            g_opt.zero_grad()
            enhanced, preds, g_loss = self.g_step(chunks, orig_chunks, metadata, vvc_features)
            fake_chunks = enhanced.detach()
            self.manual_backward(g_loss)
            g_opt.step()
        elif e_train is not None:
            enhanced, preds, g_loss = self.g_step(chunks, orig_chunks, metadata, vvc_features)
            fake_chunks = enhanced.detach()
        else:
            fake_chunks = chunks
            enhanced = None

        # DISCRIMINATOR
        if d_train:
            d_opt.zero_grad()
            fake_preds, real_preds, d_loss = self.d_step(fake_chunks, orig_chunks)
            self.manual_backward(d_loss)
            d_opt.step()
        elif d_train is not None:
            fake_preds, real_preds, d_loss = self.d_step(fake_chunks, orig_chunks)
        else:
            real_preds = None
            fake_preds = preds

        if batch_idx % 1000 == 0:
            self.log_images(
                enhanced,
                chunks,
                orig_chunks,
                fake_preds,
                real_preds,
            )

    def on_train_epoch_end(self):
        schs = self.lr_schedulers()
        if schs is None:
            return
        if not isinstance(schs, (tuple, list)):
            schs = [schs]
        for sch in schs:
            sch.step()

    def _unpack_batch_for_eval(self, batch):
        """
        Uniwersalne rozpakowywanie batcha.
        Obsługuje:
        - 5 elementów: (chunks, orig, meta, obj, features) -> Chunks PT
        - 4 elementy: (chunks, orig, meta, obj) -> FullFrame (stary styl)
        - 5 elementów: (chunks, orig, meta, obj, features) -> FullFrame PT (nowy styl)
        """
        if len(batch) == 5:
            chunks, orig_chunks, metadata, chunk_objs, vvc_features = batch
        elif len(batch) == 4:
            chunks, orig_chunks, metadata, chunk_objs = batch
            vvc_features = None
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
            
        return chunks, orig_chunks, metadata, chunk_objs, vvc_features

    def validation_step(self, batch, batch_idx):
        chunks, orig_chunks, metadata, _, vvc_features = self._unpack_batch_for_eval(batch)
        preds = None

        if self.mode != TrainingMode.DISCRIMINATOR:
            enhanced, preds, g_loss = self.g_step(
                chunks, orig_chunks, metadata, vvc_features, "val"
            )
            fake_chunks = enhanced.detach()
        else:
            fake_chunks = chunks
            enhanced = None

        if self.mode != TrainingMode.ENHANCER:
            fake_preds, real_preds, d_loss = self.d_step(
                fake_chunks, orig_chunks, "val"
            )
        else:
            real_preds = None
            fake_preds = preds

        if batch_idx % 1000 == 0:
            self.log_images(
                enhanced,
                chunks,
                orig_chunks,
                fake_preds,
                real_preds,
                "val",
            )

    def test_step(self, batch, batch_idx):
        chunks, orig_chunks, metadata, _, vvc_features = self._unpack_batch_for_eval(batch)
        preds = None

        if self.mode != TrainingMode.DISCRIMINATOR:
            enhanced, preds, g_loss = self.g_step(
                chunks, orig_chunks, metadata, vvc_features, "test"
            )
            fake_chunks = enhanced.detach()
        else:
            fake_chunks = chunks
            enhanced = None

        if self.mode != TrainingMode.ENHANCER:
            fake_preds, real_preds, d_loss = self.d_step(
                fake_chunks, orig_chunks, "test"
            )
        else:
            real_preds = None
            fake_preds = preds

        if batch_idx % 10 == 0:
            self.log_images(
                enhanced,
                chunks,
                orig_chunks,
                fake_preds,
                real_preds,
                "test",
            )

    def predict_step(self, batch, batch_idx):
        chunks, _, metadata, chunk_objs, vvc_features = self._unpack_batch_for_eval(batch)
        enhanced = self(chunks, metadata, vvc_features)

        save_root = getattr(self.config, "saved_chunk_folder", None)
        if save_root is None:
            # nic nie zapisujemy, ale zwracamy wyniki
            return enhanced.detach().cpu()

        os.makedirs(save_root, exist_ok=True)

        if self.test_full_frames:
            # zapis pełnych klatek zgodnie z FrameDataset.save_frame
            for i, frame_data in enumerate(enhanced):
                meta_i = [
                    c[i].cpu() if hasattr(c[i], "cpu") else c[i]
                    for c in chunk_objs
                ]
                FrameDataset.save_frame(
                    meta_i, frame_data.cpu().numpy(), save_root
                )
        else:
            # prosty zapis jako .npy – możesz później dopiąć dokładne nazwy jak u Piotra
            for i, chunk_data in enumerate(enhanced):
                out_path = os.path.join(
                    save_root, f"chunk_{batch_idx:05d}_{i:03d}.npy"
                )
                np.save(out_path, chunk_data.detach().cpu().numpy())

        return None

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.enhancer.parameters(),
            lr=self.enhancer_lr,
            betas=self.betas,
            weight_decay=self.enhancer_lr / 10,
        )
        opt_d = torch.optim.SGD(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            momentum=self.momentum,
            weight_decay=self.discriminator_lr / 10,
        )

        lr_schedulers = []

        if self.config.enhancer_scheduler is True:
            lr_schedulers.append(
                {
                    "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                        opt_g,
                        milestones=self.config.enhancer_scheduler_milestones,
                        gamma=self.config.enhancer_scheduler_gamma,
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                }
            )

        if self.config.discriminator_scheduler is True:
            lr_schedulers.append(
                {
                    "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                        opt_d,
                        milestones=self.config.discriminator_scheduler_milestones,
                        gamma=self.config.discriminator_scheduler_gamma,
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                }
            )

        return [opt_g, opt_d], lr_schedulers

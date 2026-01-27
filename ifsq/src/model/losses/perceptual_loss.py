import torch
from torch import nn
import torch.nn.functional as F
from .lpips import LPIPS
from einops import rearrange
from .discriminator import NLayerDiscriminator2D, weights_init


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = torch.mean(F.relu(1.0 - logits_real), dim=[1, 2, 3])
    loss_fake = torch.mean(F.relu(1.0 + logits_fake), dim=[1, 2, 3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)


class LPIPSWithDiscriminator2D(nn.Module):
    def __init__(
        self,
        disc_start,
        kl_weight=1.0,
        perceptual_weight=1.0,
        disc_num_layers=4,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
        distill_weight=1.0,
        loss_type: str = "l1",
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.distill_weight = distill_weight
        if perceptual_weight > 0:
            self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator2D(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.loss_func = l1 if loss_type == "l1" else l2

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        layer = last_layer if last_layer is not None else self.last_layer[0]

        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, 1e6).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        loss_break=None,
        split="train",
        last_layer=None,
        student_latents=None,
        teacher_latents=None,
    ):
        bs = inputs.shape[0]
        t = inputs.shape[2]
        if optimizer_idx == 0:  # Generator
            rec_loss = self.loss_func(inputs, reconstructions)
            nll_loss = rec_loss.clone()
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                nll_loss = nll_loss + self.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor(0.0)
            nll_loss = torch.mean(nll_loss)

            if posteriors is not None:
                kl_loss = posteriors.kl()
                kl_loss = torch.mean(kl_loss)
            else:
                kl_loss = torch.tensor(0.0)

            if teacher_latents is not None:
                assert student_latents is not None
                distill_loss = torch.mean(l1(teacher_latents, student_latents))
            else:
                distill_loss = torch.tensor(0.0)

            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            if global_step >= self.discriminator_iter_start:
                if self.disc_factor > 0.0:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = (
                nll_loss
                + self.kl_weight * kl_loss
                + d_weight * disc_factor * g_loss
                + self.distill_weight * distill_loss
            )
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/lpips_loss".format(split): p_loss.detach().mean(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/distill_loss".format(split): distill_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            return loss, log
        elif optimizer_idx == 1:  # Discriminator
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log


class LFQLPIPSWithDiscriminator2D(nn.Module):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        kl_weight=1.0,
        perceptual_weight=1.0,
        disc_num_layers=4,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        commit_weight=0.25,
        codebook_enlarge_ratio=3,
        codebook_enlarge_steps=2000,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
        distill_weight=1.0,
        loss_type: str = "l1",
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = 0.0
        self.codebook_weight = codebook_weight
        self.distill_weight = distill_weight
        if perceptual_weight > 0:
            self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.commit_weight = commit_weight
        self.codebook_enlarge_ratio = codebook_enlarge_ratio
        self.codebook_enlarge_steps = codebook_enlarge_steps

        self.discriminator = NLayerDiscriminator2D(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.loss_func = l1 if loss_type == "l1" else l2

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        layer = last_layer if last_layer is not None else self.last_layer[0]

        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, 1e6).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        loss_break=None,
        split="train",
        last_layer=None,
        student_latents=None,
        teacher_latents=None,
    ):
        bs = inputs.shape[0]
        t = inputs.shape[2]
        if optimizer_idx == 0:  # Generator
            rec_loss = self.loss_func(inputs, reconstructions)
            nll_loss = rec_loss.clone()
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                nll_loss = nll_loss + self.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor(0.0)
            nll_loss = torch.mean(nll_loss)

            if posteriors is not None:
                kl_loss = posteriors.kl()
                kl_loss = torch.mean(kl_loss)
            else:
                kl_loss = torch.tensor(0.0)

            if teacher_latents is not None:
                assert student_latents is not None
                distill_loss = torch.mean(l1(teacher_latents, student_latents))
            else:
                distill_loss = torch.tensor(0.0)

            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            if global_step >= self.discriminator_iter_start:
                if self.disc_factor > 0.0:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            scale_codebook_loss = (
                self.codebook_weight * loss_break.entropy_aux_loss
            )  # entropy_loss
            if self.codebook_enlarge_ratio > 0:
                scale_codebook_loss = (
                    self.codebook_enlarge_ratio
                    * (max(0, 1 - global_step / self.codebook_enlarge_steps))
                    * scale_codebook_loss
                    + scale_codebook_loss
                )

            loss = (
                nll_loss
                + self.kl_weight * kl_loss
                + d_weight * disc_factor * g_loss
                + self.distill_weight * distill_loss
                + scale_codebook_loss
                + loss_break.commitment * self.commit_weight
            )
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/per_sample_entropy".format(
                    split
                ): loss_break.per_sample_entropy.detach(),
                "{}/codebook_entropy".format(
                    split
                ): loss_break.codebook_entropy.detach(),
                "{}/commit_loss".format(split): loss_break.commitment.detach(),
                "{}/lpips_loss".format(split): p_loss.detach().mean(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/distill_loss".format(split): distill_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            return loss, log
        elif optimizer_idx == 1:  # Discriminator
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log


class VQLPIPSWithDiscriminator2D(nn.Module):
    def __init__(
        self,
        disc_start,
        codebook_weight=1.0,
        kl_weight=1.0,
        perceptual_weight=1.0,
        disc_num_layers=4,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
        distill_weight=1.0,
        loss_type: str = "l1",
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = 0.0
        self.codebook_weight = codebook_weight
        self.distill_weight = distill_weight
        if perceptual_weight > 0:
            self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator2D(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.loss_func = l1 if loss_type == "l1" else l2

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        layer = last_layer if last_layer is not None else self.last_layer[0]

        nll_grads = torch.autograd.grad(nll_loss, layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-6)
        d_weight = torch.clamp(d_weight, 0.0, 1e6).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        loss_break=None,
        split="train",
        last_layer=None,
        student_latents=None,
        teacher_latents=None,
    ):
        bs = inputs.shape[0]
        t = inputs.shape[2]
        if optimizer_idx == 0:  # Generator
            rec_loss = self.loss_func(inputs, reconstructions)
            nll_loss = rec_loss.clone()
            if self.perceptual_weight > 0:
                p_loss = self.perceptual_loss(inputs, reconstructions)
                nll_loss = nll_loss + self.perceptual_weight * p_loss
            else:
                p_loss = torch.tensor(0.0)
            nll_loss = torch.mean(nll_loss)

            if posteriors is not None:
                kl_loss = posteriors.kl()
                kl_loss = torch.mean(kl_loss)
            else:
                kl_loss = torch.tensor(0.0)

            if teacher_latents is not None:
                assert student_latents is not None
                distill_loss = torch.mean(l1(teacher_latents, student_latents))
            else:
                distill_loss = torch.tensor(0.0)

            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            if global_step >= self.discriminator_iter_start:
                if self.disc_factor > 0.0:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.tensor(1.0)
            else:
                d_weight = torch.tensor(0.0)
                g_loss = torch.tensor(0.0, requires_grad=True)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            scale_codebook_loss = self.codebook_weight * loss_break[-1]  # codebook_loss
            loss = (
                nll_loss
                + self.kl_weight * kl_loss
                + d_weight * disc_factor * g_loss
                + self.distill_weight * distill_loss
                + scale_codebook_loss
            )
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/codebook_loss".format(split): loss_break[-1].detach(),
                "{}/lpips_loss".format(split): p_loss.detach().mean(),
                "{}/kl_loss".format(split): kl_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): rec_loss.detach().mean(),
                "{}/distill_loss".format(split): distill_loss.detach().mean(),
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            return loss, log
        elif optimizer_idx == 1:  # Discriminator
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log

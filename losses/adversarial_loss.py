import torch
import torch.nn as nn
import torch.nn.functional as F


class LSGANLoss(nn.Module):
    """
    Least Squares GAN (LSGAN) Adversarial Loss Module.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward_g(self, d_fake):
        target_real = torch.ones_like(d_fake)
        return 0.5 * self.mse(d_fake, target_real)

    def forward_d(self, d_real, d_fake):
        target_real = torch.ones_like(d_real)
        target_fake = torch.zeros_like(d_fake)

        loss_real = 0.5 * self.mse(d_real, target_real)
        loss_fake = 0.5 * self.mse(d_fake, target_fake)

        return loss_real + loss_fake


class RelativisticGANLoss(nn.Module):
    """
    Relativistic LSGAN (RaGAN) Adversarial Loss Module.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward_g(self, d_real, d_fake):
        target_real = torch.ones_like(d_fake)
        target_fake = -torch.ones_like(d_real)

        d_fake_rel = d_fake - torch.mean(d_real.detach())
        d_real_rel = d_real - torch.mean(d_fake.detach())

        loss_g_fake = 0.5 * self.mse(d_fake_rel, target_real)
        loss_g_real = 0.5 * self.mse(d_real_rel, target_fake)

        return loss_g_fake + loss_g_real

    def forward_d(self, d_real, d_fake):
        target_real = torch.ones_like(d_real)
        target_fake = -torch.ones_like(d_fake)

        d_real_rel = d_real - torch.mean(d_fake.detach())
        d_fake_rel = d_fake - torch.mean(d_real.detach())

        loss_d_real = 0.5 * self.mse(d_real_rel, target_real)
        loss_d_fake = 0.5 * self.mse(d_fake_rel, target_fake)

        return loss_d_real + loss_d_fake


class DualDomainAdversarialLoss(nn.Module):
    """
    Dual-Domain Spatial & Frequency Adversarial Loss Wrapper.
    
    Formula:
    L_adv_total = lambda_adv_spatial * L_adv_spatial + lambda_adv_frequency * L_adv_frequency
    """
    def __init__(self, gan_type="lsgan", lambda_spatial=0.005, lambda_freq=0.005):
        super().__init__()
        self.lambda_spatial = lambda_spatial
        self.lambda_freq = lambda_freq

        if gan_type.lower() == "ragan":
            self.gan_loss = RelativisticGANLoss()
            self.is_relativistic = True
        else:
            self.gan_loss = LSGANLoss()
            self.is_relativistic = False

    def forward_g(self, d_spatial_real, d_spatial_fake, d_freq_real, d_freq_fake):
        """
        Calculates combined Generator adversarial loss across Spatial and Frequency domains.
        """
        if self.is_relativistic:
            loss_g_spatial = self.gan_loss.forward_g(d_spatial_real, d_spatial_fake)
            loss_g_freq = self.gan_loss.forward_g(d_freq_real, d_freq_fake)
        else:
            loss_g_spatial = self.gan_loss.forward_g(d_spatial_fake)
            loss_g_freq = self.gan_loss.forward_g(d_freq_fake)

        total_loss_g = (
            self.lambda_spatial * loss_g_spatial +
            self.lambda_freq * loss_g_freq
        )

        loss_dict = {
            "loss_g_adv_total": total_loss_g.item(),
            "loss_g_adv_spatial": loss_g_spatial.item(),
            "loss_g_adv_freq": loss_g_freq.item()
        }
        return total_loss_g, loss_dict

    def forward_d(self, d_spatial_real, d_spatial_fake, d_freq_real, d_freq_fake):
        """
        Calculates Discriminators adversarial loss across Spatial and Frequency domains.
        """
        loss_d_spatial = self.gan_loss.forward_d(d_spatial_real, d_spatial_fake)
        loss_d_freq = self.gan_loss.forward_d(d_freq_real, d_freq_fake)

        total_loss_d = loss_d_spatial + loss_d_freq

        loss_dict = {
            "loss_d_adv_total": total_loss_d.item(),
            "loss_d_spatial": loss_d_spatial.item(),
            "loss_d_freq": loss_d_freq.item()
        }
        return total_loss_d, loss_dict

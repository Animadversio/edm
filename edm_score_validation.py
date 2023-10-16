import torch
import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

def edm_x_t_traj(xT, mu, U, Lambda, sigma_ts, sigma_T=None):
    """

    Args:
        xT: B x ndim
        sigma_ts: Tdim
        mu: ndim
        U: ndim x rdim
        Lambda: rdim

    Returns:

    """
    if sigma_T is None:
        sigma_T = sigma_ts.max()
    xT_rel = xT - mu[None, :]  # B x ndim
    xT_coef = xT_rel @ U  # B x rdim
    if U.shape[1] < U.shape[0]:
        xT_residue = xT_rel - xT_coef @ U.T  # B x ndim
    else:
        xT_residue = None
    scaling_coef = torch.sqrt(sigma_ts[None, :] ** 2 + Lambda[:, None])  # rdim x Tdim
    scaling_coef = scaling_coef / torch.sqrt(sigma_T ** 2 + Lambda)[:, None]  # rdim x Tdim
    xt_scaled_coef = torch.einsum('br,rT->Tbr', xT_coef, scaling_coef)  # Tdim x B x rdim
    xt_traj_onmanif = torch.einsum('Tbr,rn->Tbn', xt_scaled_coef, U.T)  # Tdim x B x ndim
    if xT_residue is not None:
        residue_scaling = sigma_ts / sigma_T  # Tdim
        residue_traj = torch.einsum('bn,T->Tbn', xT_residue, residue_scaling, )  # B x Tdim x ndim
        xt_traj = residue_traj + xt_traj_onmanif + mu[None, None, ]  # B x Tdim x ndim
    else:
        xt_traj = xt_traj_onmanif + mu[None, None, ]
    return xt_traj



def edm_sampler_return(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_traj = []
    denoised_traj = []
    x_next = latents.to(torch.float64) * t_steps[0]
    x_traj.append(x_next.detach().clone())
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        x_traj.append(x_next.detach().clone())
        denoised_traj.append(denoised.detach().clone())

    return x_next, t_steps, x_traj, denoised_traj


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


#%%
import pandas as pd
from os.path import join
from core.utils.plot_utils import saveallforms
#%%
def edm_gaussian_score(xT, mu, U, Lambda, sigma, ):
    """
    Args:
        xT: B x ndim
        mu: ndim
        U: ndim x rdim
        Lambda: rdim
        sigma: scalar

    Returns:
        score
    """
    xT_rel = xT - mu[None, :]  # B x ndim
    xT_coef = xT_rel @ U  # B x rdim
    scaling_coef = Lambda / (sigma ** 2 + Lambda)  # rdim
    x_onmanif = (xT_coef * scaling_coef[None, :]) @ U.T  # Tdim x B x ndim
    score_x = - (xT_rel - x_onmanif) / sigma ** 2  # B x ndim
    return score_x


def edm_gaussian_1storder_score(xT, mu, U, Lambda, sigma, ):
    """
    Args:
        xT: B x ndim
        sigma_ts: Tdim
        mu: ndim
        U: ndim x rdim
        Lambda: rdim

    Returns:
        score
    """
    xT_rel = xT - mu[None, :]  # B x ndim
    x_onmanif = ((xT_rel @ U) * (Lambda[None, :] / sigma ** 2)) @ U.T # Tdim x B x ndim
    score_x = - (xT_rel - x_onmanif) / sigma ** 2  # B x ndim
    return score_x


PCAdir = r"/home/binxu/DL_Projects/imgdataset_PCAs"
figdir = r"/home/binxu/DL_Projects/edm_score_validation"
#%%
device = 'cuda'
class_idx = None
outdir = 'validation'
# Load network.
network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl'
with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)
#%%
data = torch.load(join(PCAdir, "CIFAR10_pca.pt"))
S, V, imgmean, cov_eigs = data["S"], data["V"], data["mean"], data["cov_eigs"]
V = V.to(device)
imgmean = imgmean.to(device)
cov_eigs = cov_eigs.to(device)
#%%
num_steps = 18
sigma_min = 0.002
sigma_max = 80
rho = 7
sigma_min = max(sigma_min, net.sigma_min)
sigma_max = min(sigma_max, net.sigma_max)
# Time step discretization.
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
#%%
batch_seeds = torch.arange(0, 512).long()  # rank_batches[0]
batch_size = len(batch_seeds)
rnd = StackedRandomGenerator(device, batch_seeds)
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
#%%
x_next = latents
#%%
print("max eigen", cov_eigs.max().item() * 4)
print("mean eigen", cov_eigs.mean().item() * 4)
print("min eigen", cov_eigs.min().item() * 4)
residual_stats = []
for sigma in t_steps[:]:
    sigma_t = sigma  # torch.tensor(5, device="cuda", dtype=torch.float64) # t_steps[1]
    x_probe = x_next * sigma_t
    denoised = net(x_probe, sigma_t, None)
    score_xt = (denoised - x_probe) / sigma_t ** 2
    score_xt_pred = edm_gaussian_score(x_probe.flatten(1), imgmean.flatten() * 2 - 1, V, cov_eigs * 4, sigma_t)
    score_xt_pred_iso = edm_gaussian_score(x_probe.flatten(1), imgmean.flatten() * 2 - 1, V, torch.zeros_like(cov_eigs) * 4, sigma_t)
    score_xt_pred_1st = edm_gaussian_1storder_score(x_probe.flatten(1), imgmean.flatten() * 2 - 1, V, cov_eigs * 4, sigma_t)
    print("Noise Sigma", sigma_t.item())
    ss_total = ((score_xt.flatten(1))**2).sum(dim=1).mean()
    residual = ((score_xt.flatten(1) - score_xt_pred)**2).sum(dim=1).mean()
    residual_gau1st = ((score_xt.flatten(1) - score_xt_pred_1st)**2).sum(dim=1).mean()
    residual_iso = ((score_xt.flatten(1) - score_xt_pred_iso)**2).sum(dim=1).mean()
    resid_ratio = residual / ss_total
    resid_ratio_gau1st = residual_gau1st / ss_total
    resid_ratio_iso = residual_iso / ss_total
    print("residual variance ratio of Gaussian model", resid_ratio.item())
    print("Residual variance ratio of Gaussian model without cov", resid_ratio_iso.item())
    print("residual variance ratio of Gaussian model with 1st expansion", resid_ratio_gau1st.item())
    residual_stats.append({"sigma": sigma_t.item(), "resid_ratio": resid_ratio.item(), "resid_ratio_iso": resid_ratio_iso.item(),
                           "resid_ratio_gau1st": resid_ratio_gau1st.item(),
                           "residual": residual.item(), "residual_iso": residual_iso.item(), "residual_gau1st": residual_gau1st.item(),
                           "ss_total": ss_total.item()
                           })

residual_df = pd.DataFrame(residual_stats)
residual_df.to_csv(join(figdir, "cifar10_score_approx_residual.csv"))
#%%
plt.figure(figsize=[4.5, 5])
# plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_gau1st"])
cov_eigs_sorted, _ = torch.sort(cov_eigs, descending=True)
std_eigs = torch.sqrt(cov_eigs_sorted*4).cpu().numpy()
for eigi in range(10):
    plt.axvline(x=std_eigs[eigi], linestyle=":", color="r", lw=1., label="sqrt top eigen" if eigi==0 else None)
plt.axvline(x=(cov_eigs.mean()*4).sqrt().cpu().numpy(),
            linestyle=":", color="k", lw=1., label="sqrt mean eigenval") #label="sqrt top eigen"
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio"], "-o",
             label="Score of Gaussian", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_iso"], "-o",
             label="Score of Data Center", alpha=0.5)
plt.legend()
plt.ylabel("Fraction of Squared Error")
plt.xlabel("noise scale $\sigma$")
plt.title("CIFAR10 score approximation")
saveallforms(figdir, "cifar10_score_approx_residual_curve")
plt.show()
#%%
print(torch.sum(cov_eigs * 4))
#%%
plt.imshow(make_grid(denoised, nrow=8).\
           detach().cpu().permute([1, 2, 0]).numpy())
plt.show()
#%%


#%%
network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl'
with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)

data = torch.load(join(PCAdir, "ffhq64_PCA.pt"))
cov_eigs, V, imgmean = data["eigval"], data["eigvec"], data["imgmean"]
V = V.to(device)
imgmean = imgmean.to(device)
cov_eigs = cov_eigs.to(device)
#%%
max_batch_size = 128
num_steps = 40
sigma_min = 0.002
sigma_max = 80
rho = 7
sigma_min = max(sigma_min, net.sigma_min)
sigma_max = min(sigma_max, net.sigma_max)
# Time step discretization.
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
#%%
batch_seeds = torch.arange(0, max_batch_size).long()  # rank_batches[0]
batch_size = len(batch_seeds)
rnd = StackedRandomGenerator(device, batch_seeds)
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
#%%
x_next = latents
#%%
print("max eigen", cov_eigs.max().item() * 4)
print("mean eigen", cov_eigs.mean().item() * 4)
print("min eigen", cov_eigs.min().item() * 4)
residual_stats = []
for sigma in t_steps[:]:
    sigma_t = sigma  # torch.tensor(5, device="cuda", dtype=torch.float64) # t_steps[1]
    x_probe = x_next * sigma_t
    denoised = net(x_probe, sigma_t, None)
    score_xt = (denoised - x_probe) / sigma_t ** 2
    score_xt_pred = edm_gaussian_score(x_probe.flatten(1), imgmean.flatten() * 2 - 1, V, cov_eigs * 4, sigma_t)
    score_xt_pred_iso = edm_gaussian_score(x_probe.flatten(1), imgmean.flatten() * 2 - 1, V, torch.zeros_like(cov_eigs) * 4, sigma_t)
    score_xt_pred_1st = edm_gaussian_1storder_score(x_probe.flatten(1), imgmean.flatten() * 2 - 1, V, cov_eigs * 4, sigma_t)
    print("Noise Sigma", sigma_t.item())
    ss_total = ((score_xt.flatten(1))**2).sum(dim=1).mean()
    residual = ((score_xt.flatten(1) - score_xt_pred)**2).sum(dim=1).mean()
    residual_gau1st = ((score_xt.flatten(1) - score_xt_pred_1st)**2).sum(dim=1).mean()
    residual_iso = ((score_xt.flatten(1) - score_xt_pred_iso)**2).sum(dim=1).mean()
    resid_ratio = residual / ss_total
    resid_ratio_gau1st = residual_gau1st / ss_total
    resid_ratio_iso = residual_iso / ss_total
    print("residual variance ratio of Gaussian model", resid_ratio.item())
    print("Residual variance ratio of Gaussian model without cov", resid_ratio_iso.item())
    print("residual variance ratio of Gaussian model with 1st expansion", resid_ratio_gau1st.item())
    residual_stats.append({"sigma": sigma_t.item(), "resid_ratio": resid_ratio.item(), "resid_ratio_iso": resid_ratio_iso.item(),
                           "resid_ratio_gau1st": resid_ratio_gau1st.item(),
                           "residual": residual.item(), "residual_iso": residual_iso.item(), "residual_gau1st": residual_gau1st.item(),
                           "ss_total": ss_total.item()
                           })

residual_df = pd.DataFrame(residual_stats)
residual_df.to_csv(join(figdir, "ffhq64_score_approx_residual.csv"))
#%%
plt.figure(figsize=[4.5, 5])
cov_eigs_sorted, _ = torch.sort(cov_eigs, descending=True)
std_eigs = torch.sqrt(cov_eigs_sorted*4).cpu().numpy()
for eigi in range(10):
    plt.axvline(x=std_eigs[eigi], linestyle=":", color="r", lw=1., label="sqrt top eigen" if eigi==0 else None)
plt.axvline(x=(cov_eigs.mean()*4).sqrt().cpu().numpy(),
            linestyle=":", color="k", lw=1., label="sqrt mean eigenval") #label="sqrt top eigen"
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio"], "-o",
             label="Score of Gaussian", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_iso"], "-o",
             label="Score of Data Center", alpha=0.5)
plt.legend()
plt.ylabel("Fraction of Squared Error")
plt.xlabel("noise scale $\sigma$")
plt.title("FFHQ-64 score approximation")
saveallforms(figdir, "FFHQ64_score_approx_residual_curve")
plt.show()




#%%
network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl'
with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)

data = torch.load(join(PCAdir, "afhqv264_PCA.pt"))
cov_eigs, V, imgmean = data["eigval"], data["eigvec"], data["imgmean"]
V = V.to(device)
imgmean = imgmean.to(device)
cov_eigs = cov_eigs.to(device)
#%%
max_batch_size = 128
num_steps = 40
sigma_min = 0.002
sigma_max = 80
rho = 7
sigma_min = max(sigma_min, net.sigma_min)
sigma_max = min(sigma_max, net.sigma_max)
# Time step discretization.
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
#%%
batch_seeds = torch.arange(0, max_batch_size).long()  # rank_batches[0]
batch_size = len(batch_seeds)
rnd = StackedRandomGenerator(device, batch_seeds)
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
#%%
x_next = latents
#%%
print("max eigen", cov_eigs.max().item() * 4)
print("mean eigen", cov_eigs.mean().item() * 4)
print("min eigen", cov_eigs.min().item() * 4)
residual_stats = []
for sigma in t_steps[:]:
    sigma_t = sigma  # torch.tensor(5, device="cuda", dtype=torch.float64) # t_steps[1]
    x_probe = x_next * sigma_t
    denoised = net(x_probe, sigma_t, None)
    score_xt = (denoised - x_probe) / sigma_t ** 2
    score_xt_pred = edm_gaussian_score(x_probe.flatten(1), imgmean.flatten() * 2 - 1, V, cov_eigs * 4, sigma_t)
    score_xt_pred_iso = edm_gaussian_score(x_probe.flatten(1), imgmean.flatten() * 2 - 1, V, torch.zeros_like(cov_eigs) * 4, sigma_t)
    score_xt_pred_1st = edm_gaussian_1storder_score(x_probe.flatten(1), imgmean.flatten() * 2 - 1, V, cov_eigs * 4, sigma_t)
    print("Noise Sigma", sigma_t.item())
    ss_total = ((score_xt.flatten(1))**2).sum(dim=1).mean()
    residual = ((score_xt.flatten(1) - score_xt_pred)**2).sum(dim=1).mean()
    residual_gau1st = ((score_xt.flatten(1) - score_xt_pred_1st)**2).sum(dim=1).mean()
    residual_iso = ((score_xt.flatten(1) - score_xt_pred_iso)**2).sum(dim=1).mean()
    resid_ratio = residual / ss_total
    resid_ratio_gau1st = residual_gau1st / ss_total
    resid_ratio_iso = residual_iso / ss_total
    print("residual variance ratio of Gaussian model", resid_ratio.item())
    print("Residual variance ratio of Gaussian model without cov", resid_ratio_iso.item())
    print("residual variance ratio of Gaussian model with 1st expansion", resid_ratio_gau1st.item())
    residual_stats.append({"sigma": sigma_t.item(), "resid_ratio": resid_ratio.item(), "resid_ratio_iso": resid_ratio_iso.item(),
                           "resid_ratio_gau1st": resid_ratio_gau1st.item(),
                           "residual": residual.item(), "residual_iso": residual_iso.item(), "residual_gau1st": residual_gau1st.item(),
                           "ss_total": ss_total.item()
                           })

residual_df = pd.DataFrame(residual_stats)
residual_df.to_csv(join(figdir, "afhqv264_score_approx_residual.csv"))
#%%
plt.figure(figsize=[4.5, 5])
cov_eigs_sorted, _ = torch.sort(cov_eigs, descending=True)
std_eigs = torch.sqrt(cov_eigs_sorted*4).cpu().numpy()
for eigi in range(10):
    plt.axvline(x=std_eigs[eigi], linestyle=":", color="r", lw=1., label="sqrt top eigen" if eigi==0 else None)
plt.axvline(x=(cov_eigs.mean()*4).sqrt().cpu().numpy(),
            linestyle=":", color="k", lw=1., label="sqrt mean eigenval") #label="sqrt top eigen"
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio"], "-o",
             label="Score of Gaussian", alpha=0.5)
plt.semilogy(residual_df["sigma"], residual_df["resid_ratio_iso"], "-o",
             label="Score of Data Center", alpha=0.5)
plt.legend()
plt.ylabel("Fraction of Squared Error")
plt.xlabel("noise scale $\sigma$")
plt.title("afhqv2-64 score approximation")
saveallforms(figdir, "afhqv264_score_approx_residual_curve")
plt.show()


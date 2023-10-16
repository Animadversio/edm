import torch
import os
import re
from os.path import join
import click
import sys
sys.path.append(r"/home/binxu/Github/edm")
import tqdm
import pickle
import pickle as pkl
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from core.utils.montage_utils import make_grid_np
from core.utils.plot_utils import saveallforms

def edm_x_t_traj(xT, mu, U, Lambda, sigma_ts, sigma_T=None):
    """ Compute the analytical trajectory of x_t for a given initial condition x_T

    Args:
        xT: B x ndim tensor, a batch of initial conditions
        sigma_ts: Tdim, noise level / time discretization steps
        mu: ndim, mean of the Gaussian distribution
        U: ndim x rdim, eigenvectors of the covariance matrix
        Lambda: rdim, eigenvalues of the covariance matrix

    Returns:
        xt_traj: Tdim x B x ndim, the trajectory of x_t
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


# edm_x_t_traj(torch.randn(2, 100), torch.randn(100), torch.randn(100, 10), torch.rand(10), torch.arange(10)).shape
# edm_x_t_traj(torch.randn(2, 100), torch.randn(100), torch.randn(100, 100), torch.rand(100), torch.arange(10)).shape
#%%
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

        x_traj.append(x_next.detach().clone().cpu())
        denoised_traj.append(denoised.detach().clone().cpu())

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
from torchvision.transforms import ToTensor
from tqdm import trange
PCAdir = r"/home/binxu/DL_Projects/imgdataset_PCAs"
# data = torch.load(join(PCAdir, "CIFAR10_pca.pt"))
# S, V, imgmean, cov_eigs  = data["S"], data["V"], data["mean"], data["cov_eigs"]
# # S = S.to(device)
# V = V.to(device)
# imgmean = imgmean.to(device)
# cov_eigs = cov_eigs.to(device)
def mean_cov_from_xarr_torch(xarr):
    # xarr shape [n sample, ndim]
    # get mean and covariance of the xarr
    mu = torch.mean(xarr, dim=0)
    # Covariance
    xarr_centered = xarr - mu
    cov = torch.mm(xarr_centered.T, xarr_centered) / (xarr.shape[0] - 1)
    # eigen decomposition
    Lambda, U = torch.linalg.eigh(cov)
    # uncomment the following line to assert that eigen decomposition is correct
    # assert torch.allclose(cov, U @ torch.diag(Lambda) @ U.T)
    return mu, cov, Lambda, U
#%%
from gmm_general_diffusion_lib import gaussian_mixture_score_torch, deltaGMM_scores_torch_batch
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

#%%
from scipy.integrate import solve_ivp
def edm_exact_score_reverse_solver(score_func, xT, sigma_max=80, t_eval=None):
    sol = solve_ivp(lambda sigma, x: - sigma * score_func(x, sigma),
                    (sigma_max, 0), xT, method="RK45",
                    vectorized=True, t_eval=t_eval)
    return sol.y[:, -1], sol


def exact_edm_general_gmm_reverse_diff(mus, Us, Lambdas, xT, weights=None, t_eval=None, sigma_max=80, sigma_min=0.02,
                                       device="cuda"):
    # def _score_func(x_vec, sigma):
    #     return gaussian_mixture_score_torch(torch.from_numpy(x_vec).to(device).float()[None,:], mus, Us,
    #                                         Lambdas + sigma ** 2).cpu().numpy()[0]

    def _score_func_vec(x_vec, sigma):
        """ Vectorized function for solve_ivp
            interfacing with the torch accelerated score function
        x_vec: ndim x B numpy array
        sigma: scalar float
        """
        return gaussian_mixture_score_torch(torch.from_numpy(x_vec).to(device).float().T, mus, Us,
                                            Lambdas + sigma ** 2).cpu().numpy().T

    return edm_exact_score_reverse_solver(_score_func_vec, xT, sigma_max=sigma_max, t_eval=t_eval)


def exact_edm_delta_gmm_reverse_diff(mus, xT, t_eval=None, sigma_max=80, gmm_sigma=1E-5, device="cuda"):
    # def _score_func(x_vec, sigma):
    #     return deltaGMM_scores_torch_batch(mus, sigma,
    #                                        torch.from_numpy(x_vec, ).to(device).float()[None, :],
    #                                        device=device).cpu().numpy()[0]

    def _score_func_vec(x_vec, sigma):
        """ Vectorized function for solve_ivp
            interfacing with the torch accelerated score function
        x_vec: ndim x B numpy array
        sigma: scalar float
        """
        return deltaGMM_scores_torch_batch(mus, sigma,
                                           torch.from_numpy(x_vec, ).to(device).float().T,
                                           device=device).cpu().numpy().T

    return edm_exact_score_reverse_solver(_score_func_vec, xT, sigma_max=sigma_max, t_eval=t_eval)

def edm_general_gmm_denoiser(mus, Us, Lambdas, xt_traj, sigma_trajs, weights=None,
                                       device="cuda"):
    denoiser_traj = []
    for sigma_t, xt in zip(sigma_trajs, xt_traj):
        score_vec = gaussian_mixture_score_torch(torch.from_numpy(xt).to(device).float()[None, :], mus, Us,
                                            Lambdas + sigma_t ** 2).cpu().numpy()
        denoiser = sigma_t ** 2 * score_vec + xt[None, :]
        denoiser_traj.append(denoiser)

    denoiser_traj = np.concatenate(denoiser_traj, axis=0)
    return denoiser_traj


def edm_delta_gmm_denoiser(mus, xt_traj, sigma_trajs, device="cuda"):
    denoiser_traj = []
    for sigma_t, xt in zip(sigma_trajs, xt_traj):
        score_vec = deltaGMM_scores_torch_batch(mus, sigma_t,
                                           torch.from_numpy(xt, ).to(device).float()[None, :],
                                           device=device).cpu().numpy()
        denoiser = sigma_t ** 2 * score_vec + xt[None, :]
        denoiser_traj.append(denoiser)

    denoiser_traj = np.concatenate(denoiser_traj, axis=0)
    return denoiser_traj

#%%
def tsr2montage(images_tsr, nrow=8, padding=2, imgshape=(3, 32, 32)):
    images_actual = (images_tsr.reshape(-1, *imgshape) * 127.5 + 128).clip(0, 255).to(torch.uint8)
    return ToPILImage()(make_grid(images_actual, nrow=nrow, padding=padding))


def vecs2montage(vecs, nrow=8, padding=2, imgshape=(3, 32, 32)):
    image_arr = np.clip((vecs.reshape(-1, *imgshape)\
                         .transpose(0, 2, 3, 1) + 1) / 2, 0, 1)
    mtg = make_grid_np(list(image_arr), nrow=nrow, padding=padding)
    mtg_int = (mtg * 255).astype(np.uint8)
    mtg = ToPILImage()(mtg_int)
    return mtg

#%%
from torchvision.datasets import CIFAR10
figdir = r"/home/binxu/DL_Projects/edm_analy_traj_cmp/cifar10_uncond_vp_edm_theory"
os.makedirs(figdir, exist_ok=True)
dataset = CIFAR10(r'~/Datasets', download=True)
# load whole MNIST into a single tensor
xtsr = torch.stack([ToTensor()(img) for img, _ in dataset], dim=0)
ytsr = torch.stack([torch.tensor(label) for _, label in dataset], dim=0)
imgshape = xtsr.shape[1:]
ndim = np.prod(imgshape)
xtsr = xtsr * 2 - 1  # normalize to [-1, 1] to match the DDIM model fit to CIFAR10
#%%
mu_cls = []
cov_cls = []
Lambda_cls = []
U_cls = []
weights = []
for label in trange(10):
    print(f"computing mean cov of label={label} N={(ytsr == label).sum().item()}")
    xarr = xtsr[ytsr == label, :, :].flatten(1).cuda()
    mu, cov, Lambda, U = mean_cov_from_xarr_torch(xarr)
    assert torch.allclose(cov, U @ torch.diag(Lambda) @ U.T, atol=1E-4)
    mu_cls.append(mu)
    cov_cls.append(cov)
    Lambda_cls.append(Lambda)
    U_cls.append(U)
    weights.append(xarr.shape[0])

mu_cls = torch.stack(mu_cls, axis=0)
cov_cls = torch.stack(cov_cls, axis=0)
Lambda_cls = torch.stack(Lambda_cls, axis=0)
U_cls = torch.stack(U_cls, axis=0)
# single Gaussian approximation
xarr_all = xtsr.flatten(1).cuda()
mu_all, cov_all, Lambda_all, U_all = mean_cov_from_xarr_torch(xarr_all)
#%%
device = 'cuda'
network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl'
with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)
#%%
num_steps = 18
sigma_min = 0.002
sigma_max = 80
rho = 7
sigma_min = max(sigma_min, net.sigma_min)
sigma_max = min(sigma_max, net.sigma_max)
# Time step discretization.
# step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
# t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
# t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
#%%
batch_seeds = torch.arange(256, 512).long()  # rank_batches[0]
batch_size = len(batch_seeds)
rnd = StackedRandomGenerator(device, batch_seeds)
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
# sample using the EDM sampler
skip_kwargs = dict(num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
x0_edm, t_steps, x_traj_edm, denoised_traj_edm = edm_sampler_return(net, latents.cuda(), class_labels=None,
                                    randn_like=rnd.randn_like, **skip_kwargs)
x_traj_edm_tsr = torch.stack([xt.cpu() for xt in x_traj_edm], dim=0)  # [Tdim + 1, B, n chan, res, res]
denoised_edm_tsr = torch.stack([xt.cpu() for xt in denoised_traj_edm], dim=0) # [Tdim, B, n chan, res, res]
# %%
import pickle as pkl
xT = latents * t_steps[0] # scale to sigma
xT_vec = xT.flatten(1)
xT_vec_np = xT_vec.cpu().numpy()
t_steps_np = t_steps.cpu().numpy()[:-1]
for irnd in trange(batch_size):
    RNDseed = batch_seeds[irnd].item()
    # EDM solution
    x_traj_edm_sample = x_traj_edm_tsr[:, irnd, :, :, :]
    x_traj_edm_sample_vec = x_traj_edm_sample.flatten(1)
    x_traj_edm_sample_vec_np = x_traj_edm_sample_vec.cpu().numpy()
    x_edm = x_traj_edm_sample[-1].cpu().numpy()
    img_edm = ((x_edm + 1) / 2).transpose(1, 2, 0)
    # Analytical version
    xT_vec_init = xT_vec_np[irnd]
    x_gauss, sols_gauss = exact_edm_general_gmm_reverse_diff(mu_all[None], U_all[None], Lambda_all[None],
                                                     xT_vec_init, t_eval=t_steps_np)
    x_gmm, sols_gmm = exact_edm_general_gmm_reverse_diff(mu_cls, U_cls, Lambda_cls,
                                                    xT_vec_init, t_eval=t_steps_np)
    x_delta, sols_delta = exact_edm_delta_gmm_reverse_diff(xarr_all, xT_vec_init, t_eval=t_steps_np)
    x_traj_gauss = sols_gauss.y[:, :].T  # .reshape(-1, *imgshape)
    x_traj_gmm = sols_gmm.y[:, :].T  # .reshape(-1, *imgshape)
    x_traj_delta = sols_delta.y[:, :].T  # .reshape(-1, *imgshape)
    #%%
    denoiser_traj_gauss = edm_general_gmm_denoiser(mu_all[None], U_all[None], Lambda_all[None], x_traj_gauss, t_steps_np, device="cuda")
    denoiser_traj_gmm = edm_general_gmm_denoiser(mu_cls, U_cls, Lambda_cls, x_traj_gmm, t_steps_np, device="cuda")
    denoiser_traj_delta = edm_delta_gmm_denoiser(xarr_all, x_traj_delta, t_steps_np, device="cuda")
    denoiser_traj_edm = denoised_edm_tsr[:, irnd]
    denoiser_traj_edm_vec_np = denoiser_traj_edm.flatten(1).cpu().numpy()
    #%%
    img_gauss = np.clip( ((x_gauss.reshape(*imgshape) + 1) / 2).transpose(1, 2, 0), 0, 1)
    img_gmm = np.clip( ((x_gmm.reshape(*imgshape) + 1) / 2).transpose(1, 2, 0), 0, 1)
    img_delta = np.clip( ((x_delta.reshape(*imgshape) + 1) / 2).transpose(1, 2, 0), 0, 1)
    #%%
    gmm_edm_mse = ((x_traj_gmm - x_traj_edm_sample_vec_np[:-1])**2).mean(axis=1)
    gauss_edm_mse = ((x_traj_gauss - x_traj_edm_sample_vec_np[:-1])**2).mean(axis=1)
    delta_edm_mse = ((x_traj_delta - x_traj_edm_sample_vec_np[:-1])**2).mean(axis=1)
    #%%
    denoiser_gmm_edm_mse = ((denoiser_traj_gmm - denoiser_traj_edm_vec_np)**2).mean(axis=1)
    denoiser_gauss_edm_mse = ((denoiser_traj_gauss - denoiser_traj_edm_vec_np)**2).mean(axis=1)
    denoiser_delta_edm_mse = ((denoiser_traj_delta - denoiser_traj_edm_vec_np)**2).mean(axis=1)
    #%%
    # save necessary data to reproduce results
    data = {
            "xT_init": xT_vec_init,
            "t_traj": t_steps_np,
            "x_traj_edm": x_traj_edm_sample_vec_np,
            "x_traj_gauss": x_traj_gauss,
            "x_traj_gmm": x_traj_gmm,
            "x_traj_delta": x_traj_delta,
            "denoiser_traj_gauss": denoiser_traj_gauss,
            "denoiser_traj_gmm": denoiser_traj_gmm,
            "denoiser_traj_delta": denoiser_traj_delta,
            "denoiser_traj_edm": denoiser_traj_edm_vec_np,
            "img_edm": img_edm,
            "img_gauss": img_gauss,
            "img_gmm": img_gmm,
            "img_delta": img_delta,
            "RND": RNDseed,
            }
    MSEstats = {"gmm_edm_mse": gmm_edm_mse,
                "gauss_edm_mse": gauss_edm_mse,
                "delta_edm_mse": delta_edm_mse,
                "denoiser_gmm_edm_mse": denoiser_gmm_edm_mse,
                "denoiser_gauss_edm_mse": denoiser_gauss_edm_mse,
                "denoiser_delta_edm_mse": denoiser_delta_edm_mse,
                "t_traj": t_steps_np,
                }
    pkl.dump(data, open(join(figdir, f"RND{RNDseed:03d}_edm_analy_traj.pkl"), "wb"))
    pkl.dump(MSEstats, open(join(figdir, f"RND{RNDseed:03d}_edm_analy_mse.pkl"), "wb"))
    #%%
    plt.figure(figsize=(4.5, 4))
    plt.plot(t_steps_np, gauss_edm_mse, label="Gauss")
    plt.plot(t_steps_np, gmm_edm_mse, label="GMM")
    plt.plot(t_steps_np, delta_edm_mse, label="Delta")
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("sigma")
    plt.title(f"X t state deviation MSE (RND{RNDseed:03d})")
    saveallforms(figdir, f"RND{RNDseed:03d}_edm_mse_xt_traj", )
    plt.show()
    #%%
    plt.figure(figsize=(4.5, 4))
    plt.plot(t_steps_np, denoiser_gauss_edm_mse, label="Gauss")
    plt.plot(t_steps_np, denoiser_gmm_edm_mse, label="GMM")
    plt.plot(t_steps_np, denoiser_delta_edm_mse, label="Delta")
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("sigma")
    plt.title(f"X t denoiser deviation MSE (RND{RNDseed:03d} )")
    saveallforms(figdir, f"RND{RNDseed:03d}_edm_mse_denoiser_traj", )
    plt.show()
    #%%
    mtg_horz = np.clip(make_grid_np([img_edm, img_gauss, img_gmm, img_delta], nrow=4, padding=2), 0, 1)
    mtg_vert = np.clip(make_grid_np([img_edm, img_gauss, img_gmm, img_delta], nrow=1, padding=2), 0, 1)
    plt.imsave(join(figdir, f"RND{RNDseed:03d}_x0_cmp_horz.png"), mtg_horz)
    plt.imsave(join(figdir, f"RND{RNDseed:03d}_x0_cmp_vert.png"), mtg_vert)
    #%%
    mtg_denoised_gauss = vecs2montage(denoiser_traj_gauss, nrow=6, imgshape=imgshape)
    mtg_denoised_gmm = vecs2montage(denoiser_traj_gmm, nrow=6, imgshape=imgshape)
    mtg_denoised_delta = vecs2montage(denoiser_traj_delta, nrow=6, imgshape=imgshape)
    mtg_denoised_edm = tsr2montage(denoiser_traj_edm, nrow=6, imgshape=imgshape)
    mtg_denoised_gauss.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_gauss.png"))
    mtg_denoised_gmm.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_gmm.png"))
    mtg_denoised_delta.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_delta.png"))
    mtg_denoised_edm.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_edm.png"))
    #%%




#%%
figdir = r"/home/binxu/DL_Projects/edm_analy_traj_cmp/afhqv2_uncond_vp_edm_theory"
os.makedirs(figdir, exist_ok=True)
from training.dataset import ImageFolderDataset
dataset = ImageFolderDataset("/home/binxu/Datasets/afhqv2-64x64.zip")
# load whole MNIST into a single tensor
# xtsr = torch.stack([ToTensor()(img) for img, _ in dataset], dim=0) # note this is buggy!
xtsr = np.stack([(img) for img, _ in dataset], axis=0) # note this is buggy!
xtsr = torch.from_numpy(xtsr).float() / 255.0
ytsr = torch.stack([torch.tensor(label) for _, label in dataset], dim=0)
imgshape = xtsr.shape[1:]
ndim = np.prod(imgshape)
xtsr = xtsr * 2 - 1  # normalize to [-1, 1] to match the DDIM model fit to CIFAR10
#%%
plt.imshow(xtsr[0].permute(1,2,0))
# plt.imshow(xtsr[0].transpose(1,2,0))
plt.show()
#%%
# single Gaussian approximation
xarr_all = xtsr.flatten(1).cuda()
mu_all, cov_all, Lambda_all, U_all = mean_cov_from_xarr_torch(xarr_all)
cov_all = cov_all.to("cpu")
#%%
device = 'cuda'
class_idx = None
# Load network.
network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-vp.pkl'
with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)

#%%
num_steps = 40
sigma_min = 0.002
sigma_max = 80
rho = 7
sigma_min = max(sigma_min, net.sigma_min)
sigma_max = min(sigma_max, net.sigma_max)
# Time step discretization.
# step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
# t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
# t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
#%%
batch_seeds = torch.arange(256, 512).long()  # rank_batches[0]
batch_size = len(batch_seeds)
rnd = StackedRandomGenerator(device, batch_seeds)
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
# sample using the EDM sampler
skip_kwargs = dict(num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
x0_edm, t_steps, x_traj_edm, denoised_traj_edm = edm_sampler_return(net, latents.cuda(), class_labels=None,
                                    randn_like=rnd.randn_like, **skip_kwargs)
x_traj_edm_tsr = torch.stack([xt.cpu() for xt in x_traj_edm], dim=0)  # [Tdim + 1, B, n chan, res, res]
denoised_edm_tsr = torch.stack([xt.cpu() for xt in denoised_traj_edm], dim=0) # [Tdim, B, n chan, res, res]
#%%
 # function to extent the trajectory to the same length
def extrap_end_value(x_traj, n_steps):
    x_traj_ext = np.zeros((n_steps, *x_traj.shape[1:]))
    x_traj_ext[:len(x_traj), ...] = x_traj
    x_traj_ext[len(x_traj):, ...] = x_traj[-1]
    return x_traj_ext
# %%
xT = latents * t_steps[0] # scale to sigma
xT_vec = xT.flatten(1)
xT_vec_np = xT_vec.cpu().numpy()
t_steps_np = t_steps.cpu().numpy()[:-1]
for irnd in trange(batch_size):
    RNDseed = batch_seeds[irnd].item()
    # EDM solution
    x_traj_edm_sample = x_traj_edm_tsr[:, irnd, :, :, :]
    x_traj_edm_sample_vec = x_traj_edm_sample.flatten(1)
    x_traj_edm_sample_vec_np = x_traj_edm_sample_vec.cpu().numpy()
    x_edm = x_traj_edm_sample[-1].cpu().numpy()
    img_edm = ((x_edm + 1) / 2).transpose(1, 2, 0)
    # Analytical version
    xT_vec_init = xT_vec_np[irnd]
    x_gauss, sols_gauss = exact_edm_general_gmm_reverse_diff(mu_all[None], U_all[None], Lambda_all[None],
                                                     xT_vec_init, t_eval=t_steps_np)
    x_delta, sols_delta = exact_edm_delta_gmm_reverse_diff(xarr_all, xT_vec_init, t_eval=t_steps_np)
    x_traj_gauss = sols_gauss.y[:, :].T  # .reshape(-1, *imgshape)
    x_traj_delta = sols_delta.y[:, :].T  # .reshape(-1, *imgshape)
    # sometimes the solution ends earlier
    x_traj_gauss = extrap_end_value(x_traj_gauss, len(t_steps_np))
    x_traj_delta = extrap_end_value(x_traj_delta, len(t_steps_np))
    #%%
    denoiser_traj_gauss = edm_general_gmm_denoiser(mu_all[None], U_all[None], Lambda_all[None], x_traj_gauss, t_steps_np, device="cuda")
    denoiser_traj_delta = edm_delta_gmm_denoiser(xarr_all, x_traj_delta, t_steps_np, device="cuda")
    denoiser_traj_edm = denoised_edm_tsr[:, irnd]
    denoiser_traj_edm_vec_np = denoiser_traj_edm.flatten(1).cpu().numpy()
    #%%
    img_gauss = np.clip( ((x_gauss.reshape(*imgshape) + 1) / 2).transpose(1, 2, 0), 0, 1)
    img_delta = np.clip( ((x_delta.reshape(*imgshape) + 1) / 2).transpose(1, 2, 0), 0, 1)
    #%%
    gauss_edm_mse = ((x_traj_gauss - x_traj_edm_sample_vec_np[:-1])**2).mean(axis=1)
    delta_edm_mse = ((x_traj_delta - x_traj_edm_sample_vec_np[:-1])**2).mean(axis=1)
    #%%
    denoiser_gauss_edm_mse = ((denoiser_traj_gauss - denoiser_traj_edm_vec_np)**2).mean(axis=1)
    denoiser_delta_edm_mse = ((denoiser_traj_delta - denoiser_traj_edm_vec_np)**2).mean(axis=1)
    #%%
    # save necessary data to reproduce results
    data = {
            "xT_init": xT_vec_init,
            "t_traj": t_steps_np,
            "x_traj_edm": x_traj_edm_sample_vec_np,
            "x_traj_gauss": x_traj_gauss,
            "x_traj_delta": x_traj_delta,
            "denoiser_traj_gauss": denoiser_traj_gauss,
            "denoiser_traj_delta": denoiser_traj_delta,
            "denoiser_traj_edm": denoiser_traj_edm_vec_np,
            "img_edm": img_edm,
            "img_gauss": img_gauss,
            "img_delta": img_delta,
            "RND": RNDseed,
            }
    MSEstats = {
                "gauss_edm_mse": gauss_edm_mse,
                "delta_edm_mse": delta_edm_mse,
                "denoiser_gauss_edm_mse": denoiser_gauss_edm_mse,
                "denoiser_delta_edm_mse": denoiser_delta_edm_mse,
                "t_traj": t_steps_np,
                }
    pkl.dump(data, open(join(figdir, f"RND{RNDseed:03d}_edm_analy_traj.pkl"), "wb"))
    pkl.dump(MSEstats, open(join(figdir, f"RND{RNDseed:03d}_edm_analy_mse.pkl"), "wb"))
    #%%
    plt.figure(figsize=(4.5, 4))
    plt.plot(t_steps_np, gauss_edm_mse, label="Gauss")
    plt.plot(t_steps_np, delta_edm_mse, label="Delta")
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("sigma")
    plt.title(f"X t state deviation MSE (RND{RNDseed:03d})")
    saveallforms(figdir, f"RND{RNDseed:03d}_edm_mse_xt_traj", )
    plt.show()
    #%%
    plt.figure(figsize=(4.5, 4))
    plt.plot(t_steps_np, denoiser_gauss_edm_mse, label="Gauss")
    plt.plot(t_steps_np, denoiser_delta_edm_mse, label="Delta")
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("sigma")
    plt.title(f"X t denoiser deviation MSE (RND{RNDseed:03d} )")
    saveallforms(figdir, f"RND{RNDseed:03d}_edm_mse_denoiser_traj", )
    plt.show()
    #%%
    mtg_horz = np.clip(make_grid_np([img_edm, img_gauss, img_delta], nrow=3, padding=2), 0, 1)
    mtg_vert = np.clip(make_grid_np([img_edm, img_gauss, img_delta], nrow=1, padding=2), 0, 1)
    plt.imsave(join(figdir, f"RND{RNDseed:03d}_x0_cmp_horz.png"), mtg_horz)
    plt.imsave(join(figdir, f"RND{RNDseed:03d}_x0_cmp_vert.png"), mtg_vert)
    #%%
    mtg_denoised_gauss = vecs2montage(denoiser_traj_gauss, nrow=6, imgshape=imgshape)
    mtg_denoised_delta = vecs2montage(denoiser_traj_delta, nrow=6, imgshape=imgshape)
    mtg_denoised_edm = tsr2montage(denoiser_traj_edm, nrow=6, imgshape=imgshape)
    mtg_denoised_gauss.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_gauss.png"))
    mtg_denoised_delta.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_delta.png"))
    mtg_denoised_edm.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_edm.png"))
    #%%






#%%
from training.dataset import ImageFolderDataset
figdir = r"/home/binxu/DL_Projects/edm_analy_traj_cmp/ffhq_uncond_vp_edm_theory"
os.makedirs(figdir, exist_ok=True)
dataset = ImageFolderDataset("/home/binxu/Datasets/ffhq-64x64.zip")
# load whole FFHQ into a single tensor
# xtsr = torch.stack([ToTensor()(img) for img, _ in dataset], dim=0) # note this is buggy!
xtsr = np.stack([(img) for img, _ in dataset], axis=0) # note this is buggy!
xtsr = torch.from_numpy(xtsr).float() / 255.0
ytsr = torch.stack([torch.tensor(label) for _, label in dataset], dim=0)
imgshape = xtsr.shape[1:]
ndim = np.prod(imgshape)
xtsr = xtsr * 2 - 1  # normalize to [-1, 1] to match the DDIM model fit to CIFAR10
#%%
plt.imshow(xtsr[1].permute(1, 2, 0) * 0.5 + 0.5)
plt.show()
#%%
# single Gaussian approximation
xarr_all = xtsr.flatten(1).cuda()
mu_all, cov_all, Lambda_all, U_all = mean_cov_from_xarr_torch(xarr_all)
cov_all = cov_all.to("cpu")
#%%
device = 'cuda'
class_idx = None
# Load network.
network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl'
with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)

#%%
num_steps = 40
sigma_min = 0.002
sigma_max = 80
rho = 7
sigma_min = max(sigma_min, net.sigma_min)
sigma_max = min(sigma_max, net.sigma_max)
# Time step discretization.
# step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
# t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
# t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
#%%
batch_seeds = torch.arange(256, 512).long()  # rank_batches[0]
batch_size = len(batch_seeds)
rnd = StackedRandomGenerator(device, batch_seeds)
latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
# sample using the EDM sampler
skip_kwargs = dict(num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
x0_edm, t_steps, x_traj_edm, denoised_traj_edm = edm_sampler_return(net, latents.cuda(), class_labels=None,
                                    randn_like=rnd.randn_like, **skip_kwargs)
x_traj_edm_tsr = torch.stack([xt.cpu() for xt in x_traj_edm], dim=0)  # [Tdim + 1, B, n chan, res, res]
denoised_edm_tsr = torch.stack([xt.cpu() for xt in denoised_traj_edm], dim=0) # [Tdim, B, n chan, res, res]
#%%
# function to extend the trajectory to the same length
def extrap_end_value(x_traj, n_steps):
    x_traj_ext = np.zeros((n_steps, *x_traj.shape[1:]))
    x_traj_ext[:len(x_traj), ...] = x_traj
    x_traj_ext[len(x_traj):, ...] = x_traj[-1]
    return x_traj_ext
# %%
xT = latents * t_steps[0] # scale to sigma
xT_vec = xT.flatten(1)
xT_vec_np = xT_vec.cpu().numpy()
t_steps_np = t_steps.cpu().numpy()[:-1]
for irnd in trange(batch_size):
    RNDseed = batch_seeds[irnd].item()
    # EDM solution
    x_traj_edm_sample = x_traj_edm_tsr[:, irnd, :, :, :]
    x_traj_edm_sample_vec = x_traj_edm_sample.flatten(1)
    x_traj_edm_sample_vec_np = x_traj_edm_sample_vec.cpu().numpy()
    x_edm = x_traj_edm_sample[-1].cpu().numpy()
    img_edm = ((x_edm + 1) / 2).transpose(1, 2, 0)
    # Analytical version
    xT_vec_init = xT_vec_np[irnd]
    x_gauss, sols_gauss = exact_edm_general_gmm_reverse_diff(mu_all[None], U_all[None], Lambda_all[None],
                                                     xT_vec_init, t_eval=t_steps_np)
    x_delta, sols_delta = exact_edm_delta_gmm_reverse_diff(xarr_all, xT_vec_init, t_eval=t_steps_np)
    x_traj_gauss = sols_gauss.y[:, :].T  # .reshape(-1, *imgshape)
    x_traj_delta = sols_delta.y[:, :].T  # .reshape(-1, *imgshape)
    # sometimes the solution ends earlier
    x_traj_gauss = extrap_end_value(x_traj_gauss, len(t_steps_np))
    x_traj_delta = extrap_end_value(x_traj_delta, len(t_steps_np))
    #%%
    denoiser_traj_gauss = edm_general_gmm_denoiser(mu_all[None], U_all[None], Lambda_all[None], x_traj_gauss, t_steps_np, device="cuda")
    denoiser_traj_delta = edm_delta_gmm_denoiser(xarr_all, x_traj_delta, t_steps_np, device="cuda")
    denoiser_traj_edm = denoised_edm_tsr[:, irnd]
    denoiser_traj_edm_vec_np = denoiser_traj_edm.flatten(1).cpu().numpy()
    #%%
    img_gauss = np.clip( ((x_gauss.reshape(*imgshape) + 1) / 2).transpose(1, 2, 0), 0, 1)
    img_delta = np.clip( ((x_delta.reshape(*imgshape) + 1) / 2).transpose(1, 2, 0), 0, 1)
    #%%
    gauss_edm_mse = ((x_traj_gauss - x_traj_edm_sample_vec_np[:-1])**2).mean(axis=1)
    delta_edm_mse = ((x_traj_delta - x_traj_edm_sample_vec_np[:-1])**2).mean(axis=1)
    #%%
    denoiser_gauss_edm_mse = ((denoiser_traj_gauss - denoiser_traj_edm_vec_np)**2).mean(axis=1)
    denoiser_delta_edm_mse = ((denoiser_traj_delta - denoiser_traj_edm_vec_np)**2).mean(axis=1)
    #%%
    # save necessary data to reproduce results
    data = {
            "xT_init": xT_vec_init,
            "t_traj": t_steps_np,
            "x_traj_edm": x_traj_edm_sample_vec_np,
            "x_traj_gauss": x_traj_gauss,
            "x_traj_delta": x_traj_delta,
            "denoiser_traj_gauss": denoiser_traj_gauss,
            "denoiser_traj_delta": denoiser_traj_delta,
            "denoiser_traj_edm": denoiser_traj_edm_vec_np,
            "img_edm": img_edm,
            "img_gauss": img_gauss,
            "img_delta": img_delta,
            "RND": RNDseed,
            }
    MSEstats = {
                "gauss_edm_mse": gauss_edm_mse,
                "delta_edm_mse": delta_edm_mse,
                "denoiser_gauss_edm_mse": denoiser_gauss_edm_mse,
                "denoiser_delta_edm_mse": denoiser_delta_edm_mse,
                "t_traj": t_steps_np,
                }
    pkl.dump(data, open(join(figdir, f"RND{RNDseed:03d}_edm_analy_traj.pkl"), "wb"))
    pkl.dump(MSEstats, open(join(figdir, f"RND{RNDseed:03d}_edm_analy_mse.pkl"), "wb"))
    #%%
    plt.figure(figsize=(4.5, 4))
    plt.plot(t_steps_np, gauss_edm_mse, label="Gauss")
    plt.plot(t_steps_np, delta_edm_mse, label="Delta")
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("sigma")
    plt.title(f"X t state deviation MSE (RND{RNDseed:03d})")
    saveallforms(figdir, f"RND{RNDseed:03d}_edm_mse_xt_traj", )
    plt.show()
    #%%
    plt.figure(figsize=(4.5, 4))
    plt.plot(t_steps_np, denoiser_gauss_edm_mse, label="Gauss")
    plt.plot(t_steps_np, denoiser_delta_edm_mse, label="Delta")
    plt.legend()
    plt.ylabel("MSE")
    plt.xlabel("sigma")
    plt.title(f"X t denoiser deviation MSE (RND{RNDseed:03d} )")
    saveallforms(figdir, f"RND{RNDseed:03d}_edm_mse_denoiser_traj", )
    plt.show()
    #%%
    mtg_horz = np.clip(make_grid_np([img_edm, img_gauss, img_delta], nrow=3, padding=2), 0, 1)
    mtg_vert = np.clip(make_grid_np([img_edm, img_gauss, img_delta], nrow=1, padding=2), 0, 1)
    plt.imsave(join(figdir, f"RND{RNDseed:03d}_x0_cmp_horz.png"), mtg_horz)
    plt.imsave(join(figdir, f"RND{RNDseed:03d}_x0_cmp_vert.png"), mtg_vert)
    #%%
    mtg_denoised_gauss = vecs2montage(denoiser_traj_gauss, nrow=6, imgshape=imgshape)
    mtg_denoised_delta = vecs2montage(denoiser_traj_delta, nrow=6, imgshape=imgshape)
    mtg_denoised_edm = tsr2montage(denoiser_traj_edm, nrow=6, imgshape=imgshape)
    mtg_denoised_gauss.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_gauss.png"))
    mtg_denoised_delta.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_delta.png"))
    mtg_denoised_edm.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_edm.png"))
    #%%

#%% montage re export

figdir = r"/home/binxu/DL_Projects/edm_analy_traj_cmp/afhqv2_uncond_vp_edm_theory"
imgshape = (3, 64, 64)
for irnd in trange(512):
    RNDseed = irnd
    with open(join(figdir, f"RND{RNDseed:03d}_edm_analy_traj.pkl"), "rb") as f:
        data = pkl.load(f)
        denoiser_traj_delta = data["denoiser_traj_delta"]
        denoiser_traj_gauss = data["denoiser_traj_gauss"]
        denoiser_traj_edm_vec_np = data["denoiser_traj_edm"]

    mtg_denoised_gauss = vecs2montage(denoiser_traj_gauss, nrow=7, imgshape=imgshape)
    mtg_denoised_delta = vecs2montage(denoiser_traj_delta, nrow=7, imgshape=imgshape)
    mtg_denoised_edm = vecs2montage(denoiser_traj_edm_vec_np, nrow=7, imgshape=imgshape)
    mtg_denoised_gauss.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_gauss.png"))
    mtg_denoised_delta.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_delta.png"))
    mtg_denoised_edm.save(join(figdir, f"RND{RNDseed:03d}_denoised_traj_edm.png"))
    # break

#%%
plt.figure()
plt.imshow(mtg_denoised_gauss)
plt.show()





#%%
# plt.imshow(((1 + x_next[-6].permute((1, 2, 0))) / 2).clip(0, 1).cpu())
# plt.show()
plt.imshow(vecs2montage(denoiser_traj_gauss, nrow=6))
plt.show()
plt.imshow(vecs2montage(denoiser_traj_delta, nrow=6))
plt.show()
#%%
plt.imshow(ToPILImage()(make_grid((denoised_edm_tsr[:,irnd] + 1) / 2, nrow=6)))
plt.show()
#%%
# plt.imshow(make_grid_np(list(np.clip((denoiser_traj_gauss.reshape(-1, *imgshape).transpose(0, 2, 3, 1)+1)/2,0,1)), nrow=6, padding=2))
# plt.show()
#%%
# for sigma in t_steps[:]:
#     sigma_t = sigma + EPS # torch.tensor(5, device="cuda", dtype=torch.float64) # t_steps[1]
#     x_probe = x_next * sigma_t
#     denoised = net(x_probe, sigma_t, None)
#     score_xt = (denoised - x_probe) / sigma_t ** 2
#     x_probe_flatten = x_probe.flatten(1)
#     print("Noise Sigma", sigma_t.item())
#     ss_total_vec = ((score_xt.flatten(1))**2).sum(dim=1)
#     ss_total = ss_total_vec.mean()
#     model_scores = {
#         'gauss': gaussian_mixture_score_torch(x_probe_flatten, mu_all[None], U_all[None],
#                                               Lambda_all[None] + sigma_t ** 2),
#         'gmm': gaussian_mixture_score_torch(x_probe_flatten, mu_cls, U_cls, Lambda_cls + sigma_t ** 2),
#         'delta': deltaGMM_scores_torch_batch(xarr_all, sigma_t, x_probe_flatten, device="cuda", batch_size=8).cuda(),
#         'iso': edm_gaussian_score(x_probe_flatten, mu_all, U_all, torch.zeros_like(Lambda_all) * 4, sigma),
#     }
#     stats = {'sigma': sigma_t.item(), 'ss_total': ss_total.item()}
#     data = {"score_nn": score_xt.flatten(1).cpu()}
#     for model, score_pred in model_scores.items():
#         residual_vec = ((score_xt.flatten(1) - score_pred) ** 2).sum(dim=1)
#         residual = residual_vec.mean()
#         ratio = residual / ss_total
#         print(f"residual variance ratio of {model} model: {ratio.item()}")
#         stats[f"residual_{model}"] = residual.item()
#         stats[f"resid_ratio_{model}"] = ratio.item()
#         data[f"score_{model}"] = score_pred.cpu()
#         data[f"residual_{model}_vec"] = residual_vec.cpu()
#
#     residual_stats.append(stats)
#     data_all[sigma_t.item()] = data

# residual_df = pd.DataFrame(residual_stats)
# residual_df.to_csv(join(figdir, "cifar10_GMM_score_approx_residual.csv"))
#
# pkl.dump(data_all, open(join(figdir, "cifar10_GMM_score_approx.pkl"), "wb"))

#%% Scratch zone

#%%
# x_sol, sols = exact_edm_general_gmm_reverse_diff(mu_cls, U_cls, Lambda_cls, xT_vec_np[-6],
#                                                  t_eval=t_steps.cpu().numpy())
# plt.imshow(((1 + sols.y[:, -1].reshape(imgshape).transpose((1, 2, 0))) / 2).clip(0, 1))
# plt.show()
#
# #%%
# x_sol, sols = exact_edm_general_gmm_reverse_diff(mu_all[None], U_all[None], Lambda_all[None],
#                                                  xT_vec_np[-6], t_eval=t_steps.cpu().numpy())
# plt.imshow(((1 + sols.y[:, -1].reshape(imgshape).transpose((1, 2, 0))) / 2).clip(0, 1))
# plt.show()
# #%%
# x_sol, sols = exact_edm_delta_gmm_reverse_diff(xarr_all, xT_vec_np[-6], t_eval=t_steps.cpu().numpy())
# plt.imshow(((1 + sols.y[:, -1].reshape(imgshape).transpose((1, 2, 0))) / 2).clip(0, 1))
# plt.show()

import torch
import os
import re
from os.path import join
import click
import sys
sys.path.append(r"/home/binxu/Github/edm")
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from core.utils.montage_utils import make_grid_np
import pickle as pkl
from tqdm import trange, tqdm
from core.utils.plot_utils import saveallforms
#%%
# plot the MSE curves, mean with 0.05, 0.95 quantile error bar shaded.
def compute_mean_sem_quantile(x_arr, q=0.05):
    # x_arr shape [N, T]
    mean_vec = x_arr.mean(axis=0)
    sem_vec = x_arr.std(axis=0) / np.sqrt(x_arr.shape[0])
    quantile_vec = np.quantile(x_arr, q=q, axis=0)
    return mean_vec, sem_vec, quantile_vec


def plot_mse_curves(MSE_merge, t_steps, keys=None, labels=None, colors=None):
    # MSE_merge is a dict with keys as MSE names, and values as [N, T] array
    figh = plt.figure(figsize=[5, 4.5])
    if keys is None:
        keys = MSE_merge.keys()
    if labels is None:
        labels = keys
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for key, label, color in zip(keys, labels, colors):
        mse_arr = MSE_merge[key]
        mean_vec, sem_vec, quantile_vec = compute_mean_sem_quantile(mse_arr, q=[0.25, 0.75])
        plt.plot(t_steps, mean_vec, label=label, color=color)
        # plt.fill_between(t_steps, mean_vec - sem_vec, mean_vec + sem_vec, alpha=0.3)
        plt.fill_between(t_steps, quantile_vec[0, :], quantile_vec[1, :], alpha=0.3, color=color)

    plt.legend()
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.tight_layout()
    return figh


#% Plot all curves instead of just mean
def plot_mse_curves_with_indiv(MSE_merge, t_steps, keys=None, labels=None, colors=None):
    # MSE_merge is a dict with keys as MSE names, and values as [N, T] array
    figh = plt.figure(figsize=[5, 4.5])
    if keys is None:
        keys = MSE_merge.keys()
    if labels is None:
        labels = keys
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (key, label, color) in enumerate(zip(keys, labels, colors)):
        mse_arr = MSE_merge[key]
        mean_vec, sem_vec, quantile_vec = compute_mean_sem_quantile(mse_arr, q=[0.25, 0.75])
        plt.plot(t_steps, mean_vec, label=label, lw=2, c=color)
        # plt.fill_between(t_steps, mean_vec - sem_vec, mean_vec + sem_vec, alpha=0.3)
        plt.plot(t_steps, mse_arr.T, alpha=0.05, c=color)
        # plt.fill_between(t_steps, quantile_vec[0, :], quantile_vec[1, :], alpha=0.3)
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.tight_layout()
    return figh
#%%
# create a data structure that merge the MSE stats dicts
# MSEstats = {"gmm_edm_mse": gmm_edm_mse,
#             "gauss_edm_mse": gauss_edm_mse,
#             "delta_edm_mse": delta_edm_mse,
#             "denoiser_gmm_edm_mse": denoiser_gmm_edm_mse,
#             "denoiser_gauss_edm_mse": denoiser_gauss_edm_mse,
#             "denoiser_delta_edm_mse": denoiser_delta_edm_mse,
#             "t_traj": t_steps_np,
#             }
def sweep_mse_pkl(figdir, RNDrange=(0, 512)):
    MSE_comp = {}
    for RNDseed in tqdm(RNDrange):
        mse_data = pkl.load(open(join(figdir, f"RND{RNDseed:03d}_edm_analy_mse.pkl"), "rb"))
        MSE_comp[RNDseed] = mse_data
        # pkl.load(open(join(figdir, f"RND{RNDseed:03d}_edm_analy_mse.pkl"), "rb"))
    # merge the MSE stats dicts
    MSE_merge = {}
    for key in MSE_comp[0].keys():
        MSE_merge[key] = np.stack([MSE_comp[RNDseed][key] for RNDseed in RNDrange], axis=0)
    return MSE_merge
#%%
# mean_vec, sem_vec, quantile_vec = compute_mean_sem_quantile(MSE_merge["delta_edm_mse"], q=[0.25, 0.75])
# plt.fill_between(MSE_merge["t_traj"][0], mean_vec - sem_vec, mean_vec + sem_vec, alpha=0.3)
plt.fill_between(MSE_merge["t_traj"][0], quantile_vec[0, :], quantile_vec[1, :], alpha=0.3)
plt.show()
#%% CIFAR10
sumdir = r"/home/binxu/DL_Projects/edm_analy_traj_cmp/summary"
os.makedirs(sumdir, exist_ok=True)
figdir = r"/home/binxu/DL_Projects/edm_analy_traj_cmp/cifar10_uncond_vp_edm_theory"
MSE_merge = sweep_mse_pkl(figdir, RNDrange=range(0, 512))
#%%
figh = plot_mse_curves(MSE_merge, MSE_merge["t_traj"][0],
                keys=["gauss_edm_mse", "delta_edm_mse", "gmm_edm_mse"],
                labels=["Gaussian-EDM", "Delta-EDM", "GMM-EDM"],
                colors=["C0", "C2", "C3"])
figh.suptitle("MSE between EDM and analytical x t trajectory")
saveallforms(sumdir, "cifar10_uncond_vp_edm_theory_xt_traj_MSE")
figh.show()
#%%
figh2 = plot_mse_curves(MSE_merge, MSE_merge["t_traj"][0],
                keys=["denoiser_gauss_edm_mse", "denoiser_delta_edm_mse", "denoiser_gmm_edm_mse"],
                labels=["Gaussian-EDM", "Delta-EDM", "GMM-EDM"],
                colors=["C0", "C2", "C3"])
figh2.suptitle("MSE between EDM and analytical denoiser at time t")
saveallforms(sumdir, "cifar10_uncond_vp_edm_theory_denoiser_traj_MSE")
figh2.show()
#%%
figh = plot_mse_curves_with_indiv(MSE_merge, MSE_merge["t_traj"][0],
                keys=["gauss_edm_mse", "delta_edm_mse", "gmm_edm_mse"],
                labels=["Gaussian-EDM", "Delta-EDM", "GMM-EDM"],
                colors=["C0", "C2", "C3"])
figh.suptitle("MSE between EDM and analytical x t trajectory")
saveallforms(sumdir, "cifar10_uncond_vp_edm_theory_xt_traj_MSE_indiv")
figh.show()
#%%
figh = plot_mse_curves_with_indiv(MSE_merge, MSE_merge["t_traj"][0],
                keys=["gauss_edm_mse", "gmm_edm_mse"],
                labels=["Gaussian-EDM", "GMM-EDM"],
                colors=["C0", "C3"])
figh.suptitle("MSE between EDM and analytical x t trajectory")
saveallforms(sumdir, "cifar10_uncond_vp_edm_theory_xt_traj_MSE_indiv_nodelta")
figh.show()
#%%
figh2 = plot_mse_curves_with_indiv(MSE_merge, MSE_merge["t_traj"][0],
                keys=["denoiser_gauss_edm_mse", "denoiser_delta_edm_mse", "denoiser_gmm_edm_mse"],
                labels=["Gaussian-EDM", "Delta-EDM", "GMM-EDM"],
                colors=["C0", "C2", "C3"])
figh2.suptitle("MSE between EDM and analytical denoiser at time t")
saveallforms(sumdir, "cifar10_uncond_vp_edm_theory_denoiser_traj_MSE_indiv")
figh2.show()

#%% AFHQv2
figdir = r"/home/binxu/DL_Projects/edm_analy_traj_cmp/afhqv2_uncond_vp_edm_theory"
MSE_afhq = sweep_mse_pkl(figdir, RNDrange=range(0, 512))
#%%
figh = plot_mse_curves(MSE_afhq, MSE_afhq["t_traj"][0],
                keys=["gauss_edm_mse", "delta_edm_mse"],  # "gmm_edm_mse",
                labels=["Gaussian-EDM", "Delta-EDM"],
                colors=["C0", "C2"])  # "GMM-EDM"
figh.suptitle("MSE between EDM and analytical x t trajectory")
saveallforms(sumdir, "AFHQv2_uncond_vp_edm_theory_xt_traj_MSE")
figh.show()
#%%
figh2 = plot_mse_curves(MSE_afhq, MSE_afhq["t_traj"][0],
                keys=["denoiser_gauss_edm_mse", "denoiser_delta_edm_mse"],  # "denoiser_gmm_edm_mse",
                labels=["Gaussian-EDM", "Delta-EDM"],
                colors=["C0", "C2"])  # "GMM-EDM"
figh2.suptitle("MSE between EDM and analytical denoiser at time t")
saveallforms(sumdir, "AFHQv2_uncond_vp_edm_theory_denoiser_traj_MSE")
figh2.show()

figh = plot_mse_curves_with_indiv(MSE_afhq, MSE_afhq["t_traj"][0],
                keys=["gauss_edm_mse", "delta_edm_mse"],  # "gmm_edm_mse",
                labels=["Gaussian-EDM", "Delta-EDM"],
                colors=["C0", "C2"])  # "GMM-EDM"
figh.suptitle("MSE between EDM and analytical x t trajectory")
saveallforms(sumdir, "AFHQv2_uncond_vp_edm_theory_xt_traj_MSE_indiv")
figh.show()

figh2 = plot_mse_curves_with_indiv(MSE_afhq, MSE_afhq["t_traj"][0],
                keys=["denoiser_gauss_edm_mse", "denoiser_delta_edm_mse"],  #"denoiser_gmm_edm_mse",
                labels=["Gaussian-EDM", "Delta-EDM"],
                colors=["C0", "C2"])  #"GMM-EDM"
figh2.suptitle("MSE between EDM and analytical denoiser at time t")
saveallforms(sumdir, "AFHQv2_uncond_vp_edm_theory_denoiser_traj_MSE_indiv")
figh2.show()




#%% FFHQ
figdir = r"/home/binxu/DL_Projects/edm_analy_traj_cmp/ffhq_uncond_vp_edm_theory"
MSE_ffhq = sweep_mse_pkl(figdir, RNDrange=range(0, 512))
#%%
figh = plot_mse_curves(MSE_ffhq, MSE_ffhq["t_traj"][0],
                keys=["gauss_edm_mse", "delta_edm_mse"], #"gmm_edm_mse",
                labels=["Gaussian-EDM", "Delta-EDM"],
                colors=["C0", "C2"]) #"GMM-EDM"
figh.suptitle("FFHQ64 MSE between EDM and analytical x t")
saveallforms(sumdir, "FFHQ_uncond_vp_edm_theory_xt_traj_MSE")
figh.show()
#%%
figh2 = plot_mse_curves(MSE_ffhq, MSE_ffhq["t_traj"][0],
                keys=["denoiser_gauss_edm_mse", "denoiser_delta_edm_mse"], #"denoiser_gmm_edm_mse",
                labels=["Gaussian-EDM", "Delta-EDM"],
                colors=["C0", "C2"]) #"GMM-EDM"
figh2.suptitle("FFHQ64 MSE between EDM and analytical denoiser")
saveallforms(sumdir, "FFHQ_uncond_vp_edm_theory_denoiser_traj_MSE")
figh2.show()
#%%
figh = plot_mse_curves_with_indiv(MSE_ffhq, MSE_ffhq["t_traj"][0],
                keys=["gauss_edm_mse", "delta_edm_mse"],  #"gmm_edm_mse",
                labels=["Gaussian-EDM", "Delta-EDM"],
                colors=["C0", "C2"])  #"GMM-EDM"
figh.suptitle("FFHQ64 MSE between EDM and analytical x t")
saveallforms(sumdir, "FFHQ_uncond_vp_edm_theory_xt_traj_MSE_indiv")
figh.show()
#%%
figh2 = plot_mse_curves_with_indiv(MSE_ffhq, MSE_ffhq["t_traj"][0],
                keys=["denoiser_gauss_edm_mse", "denoiser_delta_edm_mse"],  #"denoiser_gmm_edm_mse",
                labels=["Gaussian-EDM", "Delta-EDM"],
                colors=["C0", "C2"])  #"GMM-EDM"
figh2.suptitle("FFHQ64 MSE between EDM and analytical denoiser")
saveallforms(sumdir, "FFHQ_uncond_vp_edm_theory_denoiser_traj_MSE_indiv")
figh2.show()

#%%

figdir = r"/home/binxu/DL_Projects/edm_analy_traj_cmp/cifar10_uncond_vp_edm_theory"
MSE_cifar = sweep_mse_pkl(figdir, RNDrange=range(0, 512))

figdir = r"/home/binxu/DL_Projects/edm_analy_traj_cmp/afhqv2_uncond_vp_edm_theory"
MSE_afhq = sweep_mse_pkl(figdir, RNDrange=range(0, 512))

figdir = r"/home/binxu/DL_Projects/edm_analy_traj_cmp/ffhq_uncond_vp_edm_theory"
MSE_ffhq = sweep_mse_pkl(figdir, RNDrange=range(0, 512))
#%%
thresh = 0.01
for MSE_dict, label in zip([MSE_cifar, MSE_ffhq, MSE_afhq], ["CIFAR10", "FFHQ64", "AFHQv2"]):
    nsteps_below = (MSE_dict["gauss_edm_mse"].mean(axis=0)<0.01).sum()
    sigma_below = MSE_dict["t_traj"][0][nsteps_below]
    print(f"{label} Gaussian solution MSE below {thresh} for {nsteps_below} steps ({sigma_below})")
    nsteps_below = (MSE_dict["delta_edm_mse"].mean(axis=0)<0.01).sum()
    sigma_below = MSE_dict["t_traj"][0][nsteps_below]
    print(f"{label} Exact solution MSE below {thresh} for {nsteps_below} steps ({sigma_below})")
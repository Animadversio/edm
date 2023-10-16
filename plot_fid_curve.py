import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join
from core.utils.plot_utils import saveallforms
#%%
sumdir = r"D:\DL_Projects\Vision\edm_analy_sample\summary"
dataset_name = "afhqv264"
dataset_name = "cifar10"
dataset_name = "ffhq64"
for dataset_name in ["cifar10", ]:
    df = pd.read_csv(join(sumdir, f"{dataset_name}_fid_by_skipping_organized.csv"))
    if dataset_name == "cifar10":
        num_steps = 18
        sigma_min = 0.002
        sigma_max = 80
        rho = 7
    elif dataset_name == "ffhq64" or dataset_name == "afhqv264":
        num_steps = 40
        sigma_min = 0.002
        sigma_max = 80
        rho = 7
    else:
        raise NotImplementedError
    step2sigma = lambda idx: (sigma_max ** (1 / rho) + idx / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sigma2step = lambda sigma_i: (sigma_i ** (1 / rho) - sigma_max ** (1 / rho)) / (sigma_min ** (1 / rho) - sigma_max ** (1 / rho)) * (num_steps - 1)
    for i in range(num_steps):
        assert np.isclose(sigma2step(step2sigma(i)), i)
    #%%
    maxNFE = df[' NFE'].max()
    fig, ax = plt.subplots(figsize=(4.5, 5.5))
    fig.subplots_adjust(bottom=0.3)
    ax.plot(df[' NFE'], df[' FID'], 'o-')
    ax.set_xlabel("Neural Function Eval # (NFE)", fontsize=12)
    # ax.set_xlim(0, 80)
    ax.set_ylabel("FID", fontsize=12)
    ax.set_title(f"{dataset_name.upper()} EDM-hybrid FID", fontsize=14)
    ax2 = ax.secondary_xaxis(-0.15, functions=(lambda N: (maxNFE - N)//2,
                                              lambda s: maxNFE - 2*s))#functions=(time, Temp))
    ax2.set_xlabel("skipping steps", fontsize=12)
    ax3 = ax.secondary_xaxis(-0.3, functions=(lambda N: step2sigma((maxNFE - N)//2),
                                              lambda sigma: maxNFE - 2* sigma2step(sigma)))#functions=(time, Temp))
    ax3.set_xticks([2, 5, 10, 20, 30, 40, 60, 80])
    ax3.set_xlabel("skip to noise level", fontsize=12)
    saveallforms(sumdir, f"{dataset_name}_fid_by_NFE")
    plt.show()
    #%%
    maxNFE = df[' NFE'].max()
    fig, ax = plt.subplots(figsize=(4.5, 5.5))
    fig.subplots_adjust(bottom=0.3)
    ax.plot(df['Nskip'], df[' FID'], 'o-')
    ax2.set_xlabel("skipping steps", fontsize=12)
    ax.set_ylabel("FID", fontsize=12)
    ax.set_title(f"{dataset_name.upper()} EDM-hybrid FID", fontsize=14)
    ax2 = ax.secondary_xaxis(-0.15, functions=(lambda s: maxNFE - 2*s,
                                               lambda N: (maxNFE - N)//2,))#functions=(time, Temp))
    ax.set_xlabel("Neural Function Eval # (NFE)", fontsize=12)
    ax3 = ax.secondary_xaxis(-0.3, functions=(step2sigma, sigma2step))
    ax3.set_xticks([2, 5, 10, 20, 30, 40, 60, 80])
    ax3.set_xlabel("skip to noise level", fontsize=12)
    saveallforms(sumdir, f"{dataset_name}_fid_by_Nskipsteps")
    plt.show()
    #%%
    maxNFE = df[' NFE'].max()
    fig, ax = plt.subplots(figsize=(4.5, 5.5))
    fig.subplots_adjust(bottom=0.3)
    ax.plot(df[' time/noise_scale'], df[' FID'], 'o-')
    ax.set_ylabel("FID", fontsize=12)
    ax.set_xlabel("skip to noise level / time", fontsize=12)
    ax.set_title(f"{dataset_name.upper()} EDM-hybrid FID", fontsize=14)
    ax.set_xlim(0, 80)
    ax2 = ax.secondary_xaxis(-0.15, functions=(sigma2step, step2sigma))
    ax2.set_xlabel("skipping steps", fontsize=12)
    ax2.set_xticks(np.arange(0, 21, 5))
    ax3 = ax.secondary_xaxis(-0.3, functions=(lambda sigma: maxNFE - 2* sigma2step(sigma),
                                              lambda N: step2sigma((maxNFE - N)//2),))
    ax3.set_xlabel("Neural Function Eval # (NFE)", fontsize=12)
    ax3.set_xticks(np.arange(40,81,10))
    saveallforms(sumdir, f"{dataset_name}_fid_by_noiselevel")
    plt.show()
#%%
dataset_name = "cifar10"
df = pd.read_csv(join(sumdir, f"{dataset_name}_fid_by_skipping_organized.csv"))
num_steps = 18
sigma_min = 0.002
sigma_max = 80
rho = 7
step2sigma = lambda idx: (sigma_max ** (1 / rho) + idx / (num_steps - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
sigma2step = lambda sigma_i: (sigma_i ** (1 / rho) - sigma_max ** (1 / rho)) / (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho)) * (num_steps - 1)
for i in range(num_steps):
    assert np.isclose(sigma2step(step2sigma(i)), i)
#%%
maxNFE = df[' NFE'].max()
fig, ax = plt.subplots(figsize=(4.5, 5.5))
fig.subplots_adjust(bottom=0.3)
ax.plot(df[' NFE'], df[' FID'], 'o-')
ax.set_xlabel("Neural Function Eval # (NFE)", fontsize=12)
# ax.set_xlim(0, 80)
ax.set_ylabel("FID", fontsize=12)
ax.set_title(f"{dataset_name.upper()} EDM-hybrid FID", fontsize=14)
ax2 = ax.secondary_xaxis(-0.15, functions=(lambda N: (maxNFE - N)//2,
                                          lambda s: maxNFE - 2*s))#functions=(time, Temp))
ax2.set_xlabel("skipping steps", fontsize=12)
ax3 = ax.secondary_xaxis(-0.3, functions=(lambda N: step2sigma((maxNFE - N)//2),
                                          lambda sigma: maxNFE - 2* sigma2step(sigma)))#functions=(time, Temp))
ax3.set_xticks([1, 2, 5, 10, 20, 30, 40, 60, 80])
ax3.set_xlabel("skip to noise level", fontsize=12)
saveallforms(sumdir, f"{dataset_name}_fid_by_NFE")
plt.show()
#%%
maxNFE = df[' NFE'].max()
fig, ax = plt.subplots(figsize=(4.5, 5.5))
fig.subplots_adjust(bottom=0.3)
ax.plot(df['Nskip'], df[' FID'], 'o-')
ax2.set_xlabel("skipping steps", fontsize=12)
ax.set_ylabel("FID", fontsize=12)
ax.set_title(f"{dataset_name.upper()} EDM-hybrid FID", fontsize=14)
ax2 = ax.secondary_xaxis(-0.15, functions=(lambda s: maxNFE - 2*s,
                                           lambda N: (maxNFE - N)//2,))#functions=(time, Temp))
ax.set_xlabel("Neural Function Eval # (NFE)", fontsize=12)
ax3 = ax.secondary_xaxis(-0.3, functions=(step2sigma, sigma2step))
ax3.set_xticks([1, 2, 5, 10, 20, 30, 40, 60, 80])
ax3.set_xlabel("skip to noise level", fontsize=12)
saveallforms(sumdir, f"{dataset_name}_fid_by_Nskipsteps")
plt.show()
#%%
maxNFE = df[' NFE'].max()
fig, ax = plt.subplots(figsize=(4.5, 5.5))
fig.subplots_adjust(bottom=0.3)
ax.plot(df[' time/noise_scale'], df[' FID'], 'o-')
ax.set_ylabel("FID", fontsize=12)
ax.set_xlabel("skip to noise level / time", fontsize=12)
ax.set_title(f"{dataset_name.upper()} EDM-hybrid FID", fontsize=14)
ax.set_xlim(0, 80)
ax2 = ax.secondary_xaxis(-0.15, functions=(sigma2step, step2sigma))
ax2.set_xlabel("skipping steps", fontsize=12)
ax2.set_xticks(np.arange(0, 12, 2))
ax3 = ax.secondary_xaxis(-0.3, functions=(lambda sigma: maxNFE - 2* sigma2step(sigma),
                                          lambda N: step2sigma((maxNFE - N)//2),))
ax3.set_xlabel("Neural Function Eval # (NFE)", fontsize=12)
ax3.set_xticks(np.arange(15, 36, 5))
saveallforms(sumdir, f"{dataset_name}_fid_by_noiselevel")
plt.show()
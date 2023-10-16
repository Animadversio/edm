from tqdm import tqdm
import torch
import numpy as np
# import scipy.stats as stats
import matplotlib.pyplot as plt
from training.dataset import ImageFolderDataset
from dataset_tool import open_image_zip
from torch.utils.data import DataLoader
import os, io, zipfile, json, PIL
from dataset_tool import open_dest, make_transform, open_dataset, Optional, Tuple
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomAffine, ColorJitter, RandomCrop, RandomResizedCrop, CenterCrop, Pad, Lambda, Grayscale, GaussianBlur, InterpolationMode
from os.path import join
#%%
dataroot = r'E:\Datasets'
# collect all data into a nuymy array
dataset = ImageFolderDataset(join(dataroot, 'afhqv2-64x64.zip'), resolution=64, )
#%%
image_col = []
for i in tqdm(range(len(dataset))):
    image_col.append(dataset[i][0])

image_tsr = torch.from_numpy(np.array(image_col))
image_tsr = (image_tsr.to(torch.float32) / 255.0 )
image_tsr = image_tsr.cuda()
#%% compute the covariance matrix
covmat = torch.cov(image_tsr.view(image_tsr.shape[0], -1).T)
#%%
image_mean = image_tsr.mean(dim=(0))
image_kurtosis = (((image_tsr - image_mean[None])**2).sum(dim=(1, 2, 3), keepdim=True) * (image_tsr - image_mean[None])).mean(dim=0)
#%%
# eigval, eigvec = torch.eig(covmat, eigenvectors=True)
eigvals, eigvecs = torch.linalg.eigh(covmat)
#%% whiten the data set using the eigvals and eigvecs
EPS = 1E-5
image_mat = image_tsr.view(image_tsr.shape[0], -1)
image_mean = image_mat.mean(dim=(0,), keepdim=True)
image_mat_whiten = ((image_mat - image_mean) @ eigvecs) * (EPS + eigvals)**(-0.5) @ eigvecs.T
image_tsr_whiten = image_mat_whiten.view(image_tsr.shape)
#%%
plt.imshow((image_tsr_whiten[-1]).permute(1, 2, 0).cpu().numpy())
plt.show()
#%%
# save the whitened data
torch.save(image_tsr_whiten, r'E:\Datasets\afhqv2-64x64-spectral-whiten.pt')
# save the eigvals and eigvecs, image mean
torch.save((eigvals, eigvecs, image_mean), r'E:\Datasets\afhqv2-64x64-eigen.pt')
#%%
# inverse transform the whitened data
image_mat_unwhiten = image_tsr_whiten.view(image_tsr.shape[0], -1)
image_mat_unwhiten = (image_mat_unwhiten @ eigvecs) * (EPS + eigvals)**(0.5) @ eigvecs.T
image_tsr_unwhiten = (image_mat_unwhiten + image_mean).view(image_tsr.shape)
#%%
plt.imshow((image_tsr_unwhiten[-1]).permute(1, 2, 0).cpu().numpy())
plt.show()
#%%
print("Reconstruction MSE", ((image_tsr - image_tsr_unwhiten)**2).mean())  # 8.4707e-13
#%%
dataset = ImageFolderDataset(r'E:\Datasets\afhqv2-64x64.zip', resolution=64, )
# collect all data into a nuymy array
#%%
image_col = []
for i in tqdm(range(len(dataset))):
    image_col.append(dataset[i][0])

image_col = np.array(image_col)
#%%
image_arr = image_col.astype(np.float32) / 255.0
#%%
image_mean = np.mean(image_arr, axis=(0))
image_kurtosis = (((image_arr - image_mean[None])**2).sum(axis=(1, 2, 3), keepdims=True) * (image_arr - image_mean[None])).mean(axis=(0))
#%%
# compute covariance matrix
image_cov = np.cov(image_arr.reshape(image_arr.shape[0], -1).T)
print(image_cov.shape)
print(np.trace(image_cov)) # 721.348
#%%
# image_cent_cov = np.cov((image_arr - image_mean[None]).reshape(image_arr.shape[0], -1).T)
# print(image_cent_cov.shape)
# print(np.trace(image_cent_cov)) # 721.348
#%%
import matplotlib.pyplot as plt
plt.imshow((image_kurtosis.transpose(1, 2, 0) - image_kurtosis.min()) / 20, )
plt.colorbar()
plt.show()
#%%
plt.imshow(image_mean.transpose(1, 2, 0))
plt.colorbar()
plt.show()
#%%
# show eigenvalues
eigvals, eigvecs = np.linalg.eigh(image_cov)
#%%
plt.semilogy(eigvals[::-1])
plt.show()

#%%
[2, 40, 14, 9, 87, 108, 69, 22, 20, 24]
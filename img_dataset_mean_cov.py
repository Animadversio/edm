"""

"""
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, ToPILImage
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from training.dataset import ImageFolderDataset
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from os.path import join
PCAdir = r"/home/binxu/DL_Projects/imgdataset_PCAs"

#%% FFHQ-64
batch_size = 256
ffhq_dir = r"/home/binxu/Datasets/ffhq-64x64"
dataset = ImageFolderDataset(ffhq_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# example batch
imgshape = dataset[0][0].shape
imgmean = torch.zeros(np.prod(imgshape)).cuda()
imgcov = torch.zeros(np.prod(imgshape), np.prod(imgshape)).cuda()
imgbatch, _ = next(iter(dataloader))
for imgbatch, _ in tqdm.tqdm(dataloader): # 18sec
    imgbatch = imgbatch.cuda().float() / 255.0
    imgbatch = imgbatch.view(imgbatch.shape[0], -1)
    imgmean += imgbatch.sum(dim=0)
    imgcov += imgbatch.T @ imgbatch
imgmean /= len(dataset)
imgcov -= imgmean.outer(imgmean) * len(dataset)
imgcov /= len(dataset) - 1
# eigen decomposition
eigval, eigvec = torch.linalg.eigh(imgcov, )
#%%
torch.save({"eigval": eigval, "eigvec": eigvec, "imgmean": imgmean, },
           join(PCAdir, "ffhq64_PCA.pt"))
#%%
mtg = ToPILImage()(make_grid(torch.flipud(eigvec[:, -64:].t()).view(-1, *imgshape)))
mtg.save(join(PCAdir, "ffhq64_PCA_top64.png"))
#%%
plt.imshow(mtg)
plt.axis('off')
plt.tight_layout()
plt.show()
#%%
plt.semilogy(eigval.cpu().numpy()[::-1])
plt.xlabel("eigenvalue index")
plt.ylabel("eigenvalue")
plt.title("Eigenvalues of FFHQ64 dataset")
plt.savefig(join(PCAdir, "ffhq64_PCA_eigval.png"))
plt.show()


#%% AFHQv2-64
afhqv2_dir = r"/home/binxu/Datasets/afhqv2/train"
dataset = ImageFolderDataset(afhqv2_dir, )
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# example batch
imgshape = (3, 64, 64)
imgmean = torch.zeros(np.prod(imgshape)).cuda()
imgcov = torch.zeros(np.prod(imgshape), np.prod(imgshape)).cuda()
imgbatch, _ = next(iter(dataloader))
for imgbatch, _ in tqdm.tqdm(dataloader): # 18sec
    imgbatch = imgbatch.cuda().float() / 255.0
    imgbatch = Resize(64)(imgbatch)
    imgbatch = imgbatch.view(imgbatch.shape[0], -1)
    imgmean += imgbatch.sum(dim=0)
    imgcov += imgbatch.T @ imgbatch
imgmean /= len(dataset)
imgcov -= imgmean.outer(imgmean) * len(dataset)
imgcov /= len(dataset) - 1
# eigen decomposition
eigval, eigvec = torch.linalg.eigh(imgcov, )
#%%
torch.save({"eigval": eigval, "eigvec": eigvec, "imgmean": imgmean, },
           join(PCAdir, "afhqv264_PCA.pt"))
#%%
mtg = ToPILImage()(make_grid(torch.flipud(eigvec[:, -64:].t()).view(-1, *imgshape)))
mtg.save(join(PCAdir, "afhqv264_PCA_top64.png"))
#%%
plt.imshow(mtg)
plt.axis('off')
plt.tight_layout()
plt.show()
#%%
plt.semilogy(eigval.cpu().numpy()[::-1])
plt.xlabel("eigenvalue index")
plt.ylabel("eigenvalue")
plt.title("Eigenvalues of AFHQv2-64 dataset")
plt.savefig(join(PCAdir, "afhqv264_PCA_eigval.png"))
plt.show()
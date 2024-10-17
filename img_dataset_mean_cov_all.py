"""

"""
#%%
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
dataset_dir = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/EDM_datasets/datasets"
PCAdir = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/EDM_datasets/imgdataset_PCAs"

#%% CIFAR-10
batch_size = 1024
cifar10_dir = join(dataset_dir, "cifar10-32x32.zip")
dataset = ImageFolderDataset(cifar10_dir, resolution=32)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# example batch
imgshape = dataset[0][0].shape
imgmean = torch.zeros(np.prod(imgshape)).cuda()
imgcov = torch.zeros(np.prod(imgshape), np.prod(imgshape)).cuda()
imgbatch, _ = next(iter(dataloader)) # this is uint8 format, need to convert to float
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
           join(PCAdir, "cifar32_PCA.pt"))
#%%
mtg = ToPILImage()(make_grid(torch.flipud(eigvec[:, -64:].t()).view(-1, *imgshape)))
mtg.save(join(PCAdir, "cifar32_PCA_top64.png"))
#%%
plt.imshow(mtg)
plt.axis('off')
plt.tight_layout()
plt.show()
#%%
plt.semilogy(eigval.cpu().numpy()[::-1])
plt.xlabel("eigenvalue index")
plt.ylabel("eigenvalue")
plt.title("Eigenvalues of CIFAR-10-32 dataset")
plt.savefig(join(PCAdir, "cifar32_PCA_eigval.png"))



#%% FFHQ-64
batch_size = 1024
ffhq_dir = join(dataset_dir, "ffhq-64x64.zip")
dataset = ImageFolderDataset(ffhq_dir, resolution=64)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# example batch
imgshape = dataset[0][0].shape
imgmean = torch.zeros(np.prod(imgshape)).cuda()
imgcov = torch.zeros(np.prod(imgshape), np.prod(imgshape)).cuda()
imgbatch, _ = next(iter(dataloader)) # this is uint8 format, need to convert to float
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
afhqv2_dir = join(dataset_dir, "afhqv2-64x64.zip")
dataset = ImageFolderDataset(afhqv2_dir, resolution=64)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# example batch
imgshape = dataset[0][0].shape
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


#%% ImageNet-64
batch_size = 1024
INet_dir = join(dataset_dir, "imagenet-64x64.zip")
dataset = ImageFolderDataset(INet_dir, resolution=64)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# example batch
imgshape = dataset[0][0].shape
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
           join(PCAdir, "imagenet64_PCA.pt"))
#%%
mtg = ToPILImage()(make_grid(torch.flipud(eigvec[:, -64:].t()).view(-1, *imgshape)))
mtg.save(join(PCAdir, "imagenet64_PCA_top64.png"))
#%%
plt.imshow(mtg)
plt.axis('off')
plt.tight_layout()
plt.show()
#%%
plt.semilogy(eigval.cpu().numpy()[::-1])
plt.xlabel("eigenvalue index")
plt.ylabel("eigenvalue")
plt.title("Eigenvalues of imagenet-64 dataset")
plt.savefig(join(PCAdir, "imagenet64_PCA_eigval.png"))
plt.show()



#%% ImageNet-64
batch_size = 128
imagenet_dir = r"/home/binxu/Datasets/imagenet-64/train_64x64"
dataset = ImageFolderDataset(imagenet_dir, )
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
           join(PCAdir, "imagenet64_PCA.pt"))
#%%
mtg = ToPILImage()(make_grid(torch.flipud(eigvec[:, -64:].t()).view(-1, *imgshape)))
mtg.save(join(PCAdir, "imagenet64_PCA_top64.png"))
#%%
plt.imshow(mtg)
plt.axis('off')
plt.tight_layout()
plt.show()
#%%
plt.semilogy(eigval.cpu().numpy()[::-1])
plt.xlabel("eigenvalue index")
plt.ylabel("eigenvalue")
plt.title("Eigenvalues of ImageNet-64 dataset")
plt.savefig(join(PCAdir, "imagenet64_PCA_eigval.png"))
plt.show()

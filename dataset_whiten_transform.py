from dataset_tool import open_image_zip
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from training.dataset import ImageFolderDataset

# https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
def get_spectrum(image):
    fourier_image = np.fft.fftn(image)
    npix = image.shape[0]
    fourier_amplitudes = np.abs(fourier_image)**2
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins

def plot_spectrum(kvals, Abins):
    plt.loglog(kvals, Abins)
    plt.xlabel("$k$")
    plt.ylabel("$P(k)$")
    plt.tight_layout()


# whitening like: https://arxiv.org/pdf/1806.08887.pdf
def get_r(u, v):
    return np.sqrt(u**2 + v**2)

def w1(u, v, coeff=0.75):#1.0):#
    """For 1/f images, coeff should be 1, adjust to obtain flat spectra.
    coeff=0.75 works well on Butterflies dataset."""
    return get_r(u, v) ** coeff

def w2(u, v, r0=48): # r0 ~ SNR
    """Adjust r0 according to the SNR of the data, noisier data needs lower r0.
    r0=48 works well on Butterflies dataset."""
    return np.exp(- (get_r(u, v) / r0) ** 4)

def get_mask(npix=64, coeff=0.75, r0=48):
    # some parameters:
    # npix = 64  # image size
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    u = kfreq2D[0].copy()
    v = kfreq2D[1].copy()
    mask = w1(u, v, coeff) * w2(u, v, r0)
    mask[0, 0] = 1  # leave the DC component untouched
    return mask


def whiten_image(image, mean=0.45477897, std=0.29):
    img_white = image.copy()
    for c in range(3):
        # whiten channels separately
        fourier_image = np.fft.fftn(image[c])
        img_white[c] = np.fft.ifftn(fourier_image * mask).real.astype(np.float32)
    img_white -= mean
    img_white *= std
    img_white += mean
    # rescale to same range as original
    # img_white -= np.min(img_white)
    # img_white /= np.max(img_white)
    # img_white *= np.max(image) - np.min(image)
    # img_white += np.min(image)
    return img_white


def transform_whiten(img):
    """Just add to datapipeline after transforms.ToTensor()"""
    img = img.detach().cpu().numpy()
    return torch.tensor(whiten_image(img))


def inverse_whitening(img_white, mean=0.45477897, std=0.29):
    image = img_white.copy()
    for c in range(3):
        # whiten channels separately
        fourier_img_white = np.fft.fftn(img_white[c])
        image[c] = np.fft.ifftn(fourier_img_white / mask).real.astype(np.float32)
    image -= mean
    image /= std
    image += mean
    # rescale to [-1, 1]
    # image -= np.min(image)
    # image /= np.max(image)
    # image *= 2
    # image -= 1
    return image
#%%
import os, io, zipfile, json, PIL
from dataset_tool import open_dest, make_transform, open_dataset, Optional, Tuple
def transform_dataset(
    source: str,
    dest: str,
    transform_image,
    max_images: Optional[int],
    resolution: Optional[Tuple[int, int]]
):
    PIL.Image.init()

    if dest == '':
        raise Exception('--dest output filename or directory must not be an empty string')

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)

    if resolution is None: resolution = (None, None)

    dataset_attrs = None

    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
        # print(image['img'].shape)
        # print(image['img'].dtype)
        # Apply crop and resize.
        img = transform_image(image['img'])
        if img is None:
            continue

        # Error check to require uniform image attributes across
        # the whole dataset.
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {'width': img.shape[1], 'height': img.shape[0], 'channels': channels}
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                raise Exception(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3]:
                raise Exception('Input images must be stored as RGB or grayscale')
            if width != 2 ** int(np.floor(np.log2(width))):
                raise Exception('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()]
            raise Exception(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

        # Save the image as an uncompressed PNG.
        img = PIL.Image.fromarray(img, {1: 'L', 3: 'RGB'}[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, image['label']] if image['label'] is not None else None)

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()

#%%
mask = get_mask(npix=64, coeff=0.75, r0=48)
# dataset = ImageFolderDataset(r'E:\Datasets\afhqv2-64x64.zip', resolution=64)
#%%
# plt.imshow((np.clip(whiten_image(image_orig.astype(np.float32) / 255.0), 0, 1) * 255.0).astype(np.uint8).transpose(1,2,0))
# plt.show()

#%%
def whiten_image_process(image):
    return (np.clip(
        whiten_image(image.astype(np.float32).transpose(2, 0, 1) / 255.0,
                     mean=0.45477897, std=0.29),
                    0, 1) * 255.0
            ).astype(np.uint8).transpose(1, 2, 0)


def whiten_image_process_nonorm(image):
    return (np.clip(
        whiten_image(image.astype(np.float32).transpose(2, 0, 1) / 255.0,
                     mean=0, std=1),
        0, 1) * 255.0
            ).astype(np.uint8).transpose(1, 2, 0)
#%%
transform_dataset(
    source=r'E:\Datasets\afhqv2-64x64.zip',
    dest=r'E:\Datasets\afhqv2-64x64-whitened.zip',
    transform_image=whiten_image_process,
    max_images=None,
    resolution=(64, 64)
)
#%%
transform_dataset(
    source=r'E:\Datasets\afhqv2-64x64.zip',
    dest=r'E:\Datasets\afhqv2-64x64-whitened-nonorm.zip',
    transform_image=whiten_image_process_nonorm,
    max_images=None,
    resolution=(64, 64)
)
#%%
transform_dataset(
    source=r'E:\Datasets\ffhq-64x64.zip',
    dest=r'E:\Datasets\ffhq-64x64-whitened.zip',
    transform_image=whiten_image_process,
    max_images=None,
    resolution=(64, 64)
)
#%%
transform_dataset(
    source=r'E:\Datasets\ffhq-64x64.zip',
    dest=r'E:\Datasets\ffhq-64x64-whitened-nonorm.zip',
    transform_image=whiten_image_process_nonorm,
    max_images=None,
    resolution=(64, 64)
)
#%%

ImageFolderDataset(r'E:\Datasets\afhqv2-64x64-whitened.zip', resolution=64)
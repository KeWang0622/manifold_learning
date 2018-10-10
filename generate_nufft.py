#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import cv2
import sigpy as sp
import sigpy.plot as pl
# from skimage import io
import matplotlib.pyplot as plt
import os

# In[92]:


# image = io.imread('http://farm3.static.flickr.com/2506/3724084193_802ea38fc5.jpg',as_gray=True)
# resized_img=cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
# pl.ImagePlot(resized_img)
# dcf = np.load('dcf.npy')


# # In[52]:


# coord = np.load('coord.npy')
# nufft = sp.nufft(resized_img, coord)
# print(nufft.shape)
# img_reconstruct=sp.nufft_adjoint(dcf*nufft, coord)
# pl.ImagePlot(img_reconstruct)


# In[ ]:





# In[127]:


def load_images_from_folder(folder, n_im, coord, normalize=False, imrotate=False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    """

    # Initialize the arrays:
    if imrotate:  # number of images is 4 * n_im
        bigy = np.empty((n_im*4, 64, 64))
        bigx = np.empty((n_im*4, 1536, 6, 2))
    else:
        bigy = np.empty((n_im, 64, 64))
        bigx = np.empty((n_im, 1536, 6, 2))

    im = 0  # image counter

    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            bigy_temp = cv2.imread(os.path.join(folder, filename),cv2.IMREAD_GRAYSCALE)
            bigy_temp = cv2.resize(bigy_temp,(64, 64),interpolation=cv2.INTER_CUBIC)
            bigy_temp = np.float64(bigy_temp)/np.max(bigy_temp)
#         print(address)
        
            # bigy_temp=cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
            bigy[im, :, :] = bigy_temp
            bigx[im, :, :, :] = create_x_nufft(bigy_temp, coord, normalize)
            im += 1
            if imrotate:
                for angle in [90, 180, 270]:
                    bigy_rot = im_rotate(bigy_temp, angle)

                    bigy[im, :, :] = bigy_rot
                    bigx[im, :, :, :] = create_x_nufft(bigy_rot, coord, normalize)            
                    im += 1

            if imrotate:
                if im > (n_im * 4 - 1):  # how many images to load
                    break
            else:
                if im > (n_im - 1):  # how many images to load
                    break

        if normalize:
            bigx = (bigx - np.amin(bigx)) / (np.amax(bigx) - np.amin(bigx))

    return bigx, bigy


# In[ ]:





# In[128]:


def im_rotate(img, angle):
    """ Rotates an image by angle degrees
    :param img: input image
    :param angle: angle by which the image is rotated, in degrees
    :return: rotated image
    """

    rows, cols = img.shape
    rotM = cv2.getRotationMatrix2D((cols/2-0.5, rows/2-0.5), angle, 1)
    imrotated = cv2.warpAffine(img, rotM, (cols, rows))

    return imrotated


# In[131]:


def create_x_nufft(y, coord, normalize=False):
    """
    Prepares frequency data from image data: first image y is padded by 8
    pixels of value zero from each side (y_pad_loc1), then second image is
    created by moving the input image (64x64) 8 pixels down -> two same images
    at different locations are created; then both images are transformed to
    frequency space and their frequency space is combined as if the image
    moved half-way through the acquisition (upper part of freq space from one
    image and lower part of freq space from another image)
    expands the dimensions from 3D to 4D, and normalizes if normalize=True
    :param y: input image
    :param normalize: if True - the frequency data will be normalized
    :return: "Motion corrupted" frequency-space data of the input image,
    4D array of size (1, im_size1, im_size2, 2), third dimension (size: 2)
    contains real and imaginary part
    """


    nufft = sp.nufft(y, coord) #nufft
    
    x = np.dstack((nufft.real, nufft.imag))

    x = np.expand_dims(x, axis=0)
    
    if normalize:
        x = x - np.mean(x)
    return x


# In[132]:


# bigx, bigy=load_images_from_folder("pictures.txt", 30, coord,normalize=False, imrotate=True)


# # In[79]:


# print(bigx.shape)
# print(bigy.shape)


# # In[94]:


# pl.ImagePlot(sp.nufft_adjoint(dcf*(bigx[2,:,:,0]+1j*bigx[2,:,:,1]), coord))


# # In[87]:


# pl.ImagePlot(bigy[0,:,:])


# In[ ]:






# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 22:38:39 2021

@author: abc
"""

"""

NON - LOCAL MEANS DENOSING

"""

import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma

#read our image convert in into floating point value
img = img_as_float(io.imread("BSE_25sigma_noisy.jpg"))

#estimate sigma
sigma_est = np.mean(estimate_sigma(img, multichannel=False))

#define non-local means filter
denoise_img = denoise_nl_means(img, h=1.15*sigma_est, fast_mode=True, 
                               patch_size = 5,
                               patch_distance = 3, 
                               multichannel=False)

#show the images
cv2.imshow("Original images ", img)
cv2.imshow("denoise nonlocal filtered image", denoise_img)
cv2.waitKey(0)
cv2.destroyAllWindows()






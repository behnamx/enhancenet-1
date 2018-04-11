# cifar10.py ---
#
# Filename: cifar10.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sun Jan 14 20:44:24 2018 (-0800)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:

import os
import scipy.misc
import numpy as np
from math import floor
from PIL import Image

def load_data(data_dir, data_type):
    """Function to load data from CIFAR10.

    Parameters
    ----------
    data_dir : string
        Absolute path to the directory containing the extracted CIFAR10 files.

    data_type : string
        Either "train" or "test", which loads the entire train/test data in
        concatenated form.

    Returns
    -------
    data : ndarray (uint8)
        Data from the CIFAR10 dataset corresponding to the train/test
        split. The datata should be in NHWC format.

    labels : ndarray (int)
        Labels for each data. Integers ranging between 0 and 9.

    """

    if data_type == "train":
        high_res_directory = 'input_train/'
    elif data_type == "test":
        high_res_directory = 'input_test/'
    else:
        raise ValueError("Wrong data type {}".format(data_type))
        
    data = []
    label = []

    print "hello"

    # Loads images from input folder into downsampled low res, stores
    # orginal images into labels and downsampled into data
    # Code influence from http://webdav.tue.mpg.de/pixel/enhancenet/

    # Downsample High res images in output directory
    for filename in os.listdir(high_res_directory):

        # Load high res image from input directory
        img     = Image.open(os.path.join(high_res_directory, filename)).convert('RGB')         
        scale   = 4
        w, h    = img.size

        # Save high res image into Training Labels
        label   += np.array(img)/255

        # Resize high res image to 1/4 scale
        img.crop((0, 0, floor(w/scale), floor(h/scale)))
        img = img.resize((w//scale, h//scale), Image.ANTIALIAS)

        # Add low res image to Training Data
        data += np.array(img)/255

    # Concat them
    data = np.concatenate(data)
    label = np.concatenate(label)

    # Turn data into (NxHxWxC) format, so that we can easily process it, where
    # N=number of images, H=height, W=widht, C=channels. Note that this
    # corresponds to Tensorflow format that we will use later.
    data = np.transpose(np.reshape(data, (-1, 3, 32, 32)), (0, 2, 3, 1))

    return data, label


#
# cifar10.py ends here

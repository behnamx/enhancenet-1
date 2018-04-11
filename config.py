# config.py ---
#
# Filename: config.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Jan 15 12:08:58 2018 (-0800)
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

import argparse


# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")


train_arg.add_argument("--data_dir", type=str,
                       default="/Users/kwang/Downloads/cifar-10-batches-py",
                       help="Directory with CIFAR10 data")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-3,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=100,
                       help="Size of each training batch")

train_arg.add_argument("--max_iter", type=int,
                       default=5000,
                       help="Number of iterations to train")

train_arg.add_argument("--log_dir", type=str,
                       default="./logs",
                       help="Directory to save logs and current model")

train_arg.add_argument("--save_dir", type=str,
                       default="./save",
                       help="Directory to save the best model")

train_arg.add_argument("--val_freq", type=int,
                       default=500,
                       help="Validation interval")

train_arg.add_argument("--report_freq", type=int,
                       default=50,
                       help="Summary interval")



def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here

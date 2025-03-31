# /home/dbcloud/PycharmProjects/mllm4sam/app/util/utils.py
# Copyright (c) 2024, NVIDIA CORPORATION.
# All rights reserved.

import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

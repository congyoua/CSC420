import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.transforms import ColorJitter
from transformers import SegformerImageProcessor

from dataset import Dataset1, Dataset2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_datasets(task):
    set_seed(42)
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
    image_names = os.listdir("data/images")
    train_names, test_names = train_test_split(image_names, test_size=0.4, random_state=42)
    train_names, val_names = train_test_split(train_names, test_size=0.166, random_state=42)
    image_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")

    if task == 1:
        train_dataset = Dataset1.myDataset(train_names, feature_extractor=image_processor, jitter=jitter)
        valid_dataset = Dataset1.myDataset(val_names, feature_extractor=image_processor)
        test_dataset = Dataset1.myDataset(test_names, feature_extractor=image_processor)
    elif task == 2:
        train_dataset = Dataset2.myDataset(train_names, feature_extractor=image_processor, jitter=jitter)
        valid_dataset = Dataset2.myDataset(val_names, feature_extractor=image_processor)
        test_dataset = Dataset2.myDataset(test_names, feature_extractor=image_processor)
    else:
        raise ValueError("You must choose between task 1 and task 2")

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(valid_dataset))
    print("Number of test examples:", len(test_dataset))
    return train_dataset, valid_dataset, test_dataset

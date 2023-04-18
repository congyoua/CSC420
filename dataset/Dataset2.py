from torch.utils.data import Dataset
from PIL import Image
import os


class myDataset(Dataset):
    def __init__(self, image_names, feature_extractor, jitter=None):
        self.image_dir = "./data/images"
        self.label_dir = "./data/labels"
        self.feature_extractor = feature_extractor
        self.image_names = image_names
        self.jitter = jitter

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name[:-4] + "_clothes.png")
        image = Image.open(image_path)
        label = Image.open(label_path)
        if self.jitter:
            image = self.jitter(image)

        encoded_inputs = self.feature_extractor(image, label, return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs

    def get_name(self, idx):
        return self.image_names[idx]

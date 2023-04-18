import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import SegformerImageProcessor

from model import SegFormer1, SegFormer2
from utils import preprocessor


def predict(input_image, model, palette, image_processor):
    # Preprocess the input image
    encoding = image_processor(input_image, return_tensors="pt")
    pixel_values = encoding.pixel_values

    # Perform inference
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits.cpu()
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=input_image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)

        for label, color in enumerate(palette):
            color_seg[pred_seg == label, :] = color
        color_seg = color_seg[..., ::-1]  # convert to RGB

        return color_seg


def main():
    os.chdir("..")
    image_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    _, _, test_dataset = preprocessor.get_datasets(task=1)

    model = SegFormer1.get_model()
    checkpoint = torch.load("output/task1_best_model.pth")
    model.load_state_dict(checkpoint)
    model.eval()
    palette = np.array([[0, 0, 0],
                        [255, 255, 255]])
    for i in range(len(test_dataset)):
        Image.fromarray(predict(Image.open(os.path.join("data/images/", test_dataset.get_name(i))), model, palette,
                                image_processor)).save(
            os.path.join("output/test_output1", test_dataset.get_name(i)[:-4] + "_person.png"))

    _, _, test_dataset = preprocessor.get_datasets(task=2)

    model = SegFormer2.get_model()
    checkpoint = torch.load("output/task2_best_model.pth")
    model.load_state_dict(checkpoint)
    model.eval()
    palette = np.array([[0, 0, 0],
                        [0, 0, 128],
                        [0, 128, 0],
                        [0, 128, 128],
                        [128, 0, 0],
                        [128, 0, 128],
                        [120, 120, 0]])
    for i in range(len(test_dataset)):
        Image.fromarray(predict(Image.open(os.path.join("data/images/", test_dataset.get_name(i))), model, palette,
                                image_processor)).save(
            os.path.join("output/test_output2", test_dataset.get_name(i)[:-4] + "_clothes.png"))


if __name__ == "__main__":
    main()

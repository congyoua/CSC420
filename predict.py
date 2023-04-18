import argparse
import os

import numpy as np
import torch
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt
from model import SegFormer1, SegFormer2
from transformers import SegformerImageProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Perform model inference")
    parser.add_argument("-t", "--task", type=int, choices=[1, 2], required=True,
                        help="Choose between task 1 and task 2")
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("-o", "--output_path", type=str, default=None, help="Path to save the output image")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the output image using Matplotlib")
    parser.add_argument("-m", "--mask", action="store_true", help="Output image shows mask only")
    args = parser.parse_args()

    if not args.plot and args.output_path is None:
        raise ValueError("You must choose at least one of --plot and --output_path.")
    return args


def predict(input_image, model, palette, mask, image_processor):
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
        if mask:
            output_image = color_seg
        else:
            output_image = np.array(input_image) * 0.3 + color_seg * 0.7  # plot the image with the segmentation map
            output_image = output_image.astype(np.uint8)
        return output_image

def main():
    args = parse_args()
    image_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    if args.task == 1:
        model = SegFormer1.get_model()
        checkpoint = torch.load("output/task1_best_model.pth")
        model.load_state_dict(checkpoint)
        model.eval()
        palette = np.array([[0, 0, 0],
                            [255, 255, 255]])
    elif args.task == 2:
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

    if args.plot:
        try:
            input_image = Image.open(args.input_path)
        except IOError:
            raise ValueError(f"Invalid input image: {args.input_path}")
        plt.figure(figsize=(15, 10))
        plt.imshow(predict(input_image, model, palette, args.mask, image_processor))
        plt.show()
    else:
        if os.path.isdir(args.input_path) and os.path.splitext(args.output_path)[1] == '':
            for filename in os.listdir(args.input_path):
                file_path = os.path.join(args.input_path, filename)
                try:
                    input_image = Image.open(file_path)
                except IOError:
                    raise ValueError(f"Invalid input image: {file_path}")
                Image.fromarray(predict(input_image, model, palette, args.mask, image_processor)).save(
                    os.path.join(args.output_path, filename))
        elif os.path.isfile(args.input_path) and os.path.splitext(args.output_path)[1] != '':
            Image.fromarray(predict(model, palette, args.mask, image_processor)).save(
                os.path.join(args.output_path))
        else:
            raise ValueError("Input and output must have the same format (both file or both directory)")


if __name__ == "__main__":
    main()

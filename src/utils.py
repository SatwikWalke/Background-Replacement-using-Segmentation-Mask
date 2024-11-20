# utils.py

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Load the DeepLabV3 pre-trained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval()

def preprocess_image(image_path):
    input_image = Image.open(image_path)
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(input_image).unsqueeze(0)

def get_foreground_mask(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return output_predictions == 15  # Class 15 for 'person' in COCO dataset

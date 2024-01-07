import os
import torch
from torch import nn
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms import ToPILImage
import sys

from model11 import *

def image_chunks(image_path, chunk_size=128):
    img = Image.open(image_path)
    width, height = img.size
    
    chunks = []
    for i in range(0, height, chunk_size):
        for j in range(0, width, chunk_size):
            chunk = F.crop(img, i, j, chunk_size, chunk_size)
            chunks.append(torchvision.transforms.ToTensor()(chunk)[:3,:,:])
    return chunks, [height, width]

def normalize(image):
    return (image - torch.min(image)) / (torch.max(image) - torch.min(image))

def save_img_parts(images, image_size, image_name="output.png", chunk_size=128*4):
    new_size = image_size[:]
    if image_size[0]%chunk_size !=0: new_size[0] = ((image_size[0]+chunk_size)//chunk_size)*chunk_size
    if image_size[1]%chunk_size !=0: new_size[1] = ((image_size[1]+chunk_size)//chunk_size)*chunk_size

    x_new = torch.zeros((3,new_size[0], new_size[1]))
    w_axis = new_size[0] // images[0].shape[1]
    h_axis = new_size[1] // images[0].shape[2]
    
    for w in range(w_axis):
        for h in range(h_axis):
            idx = w * h_axis + h
            ww = w*chunk_size
            hh = h*chunk_size
            x_new[:, ww:ww+chunk_size, hh:hh+chunk_size] = images[idx]
    x_new = x_new[:, :image_size[0], :image_size[1]]
    print(f"upscaled size: {x_new.shape}")

    x_cat = normalize(x_new)
    image = ToPILImage()(x_cat)
    image.save(image_name)

image_path = sys.argv[1]
output_name = sys.argv[2] if len(sys.argv)==3 else None

img_chunks, size = image_chunks(image_path)
img_chunks = [x.cuda() for x in img_chunks]
size = [x*4 for x in size]

all_parts = []
with torch.no_grad():
    m = upscaler().cuda()
    m.load_state_dict(torch.load(f"models/model{11}.pt"))
    for i,image_part in enumerate(tqdm(img_chunks)):
        part_out = m(image_part.unsqueeze(0)).squeeze(0)
        all_parts.append(part_out)

save_img_parts(all_parts, size, output_name if output_name else f"{image_path}_upscaled.png")
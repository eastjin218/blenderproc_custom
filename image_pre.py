import os, glob
import numpy as np
from PIL import Image

image_path = '/home/BlenderProc/custom_data/four_img.png'
img = Image.open(image_path)
img_np = np.array(img)
img_double = np.concatenate((img_np, img_np), axis=0)
img_four = np.concatenate((img_double, img_double), axis=1)
img_raw = Image.fromarray(img_four)
img_raw.save('./img_16.png')
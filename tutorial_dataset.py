import json
import torch
import cv2
import numpy as np
import colorgram
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import os

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

class PixelsDataset(Dataset):
    def __init__(self, proportion=0.2):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.proportion = proportion

    def __len__(self):
        return len(self.data)

    def _get_pixels(self, img):
        mask = np.random.choice([0, 1], size=img.shape[:2], p=[1-self.proportion, self.proportion])

        return img * np.repeat(mask, 3, axis=1).reshape(img.shape)

    def __getitem__(self, idx):
        item = self.data[idx]

        target_filename = item['target']
        prompt = item['prompt']

        target = cv2.imread('./training/fill50k/' + target_filename)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        pixel_hint = self._get_pixels(target)

        return dict(jpg=target, txt=prompt, hint=pixel_hint)

class ColorDataset(Dataset):
    def __init__(self, n_colors=10, color_extract_res=256):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.n_colors = n_colors
        self.color_extract_res = color_extract_res

    def __len__(self):
        return len(self.data)

    def _get_color_hint(self, img):
        pil_target = Image.fromarray(img)

        if pil_target.width * pil_target.height * 3 > 256 * 256 * 3:
            pil_target = ImageOps.contain(pil_target, (self.color_extract_res,self.color_extract_res))

        colors = colorgram.extract(pil_target, self.n_colors)

        color_output = torch.zeros(self.n_colors, 4)
        
        for i, c in enumerate(colors):
            color_output[i, 0] = c.rgb.r / 255.0
            color_output[i, 1] = c.rgb.g / 255.0
            color_output[i, 2] = c.rgb.b / 255.0
            color_output[i, 3] = 1.0

        return color_output

    def __getitem__(self, idx):
        files = ['./training/fill50k/random/' + f for f in os.listdir('./training/fill50k/random/')]

        file = files[idx]
        target = cv2.imread(file)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        color_hint = self._get_color_hint(target)
        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt="lol", hint=color_hint)

# saves a new image with the colors from colorgram Color class
# aligned horizontally as bands in a 256x256 image
def save_as_img(colors, filename):
    img = Image.new('RGB', (256, 256))
    for i, c in enumerate(colors):
        img.paste(c.rgb, (i*256//len(colors), 0, (i+1)*256//len(colors), 256))
    img.save(filename)


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     dataset = PixelsDataset(proportion=0.3)
#     print(len(dataset))
#     for i in range(1):
#         hint = dataset[i]['hint']

#         plt.imshow(hint)

#         plt.savefig('lol.png')
#         print(hint.shape)



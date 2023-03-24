import webdataset as wds
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import random

def get_pixels(img, proportion):
    mask = np.random.choice([0, 1], size=img.shape[:2], p=[1-proportion, proportion])

    return img * np.repeat(mask, 3, axis=1).reshape(img.shape)

def make_parser(hint_type, proportion, side_length):

    def get_pixels_item(item):
        np_array = np.frombuffer(item['jpg'], np.uint8)
        cv2img = cv2.cvtColor(cv2.imdecode(np_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        cv2img = cv2.resize(cv2img, (side_length, side_length), interpolation=cv2.INTER_AREA) 
        # cv2.imwrite(str(j) +'.jpg', cv2img)
        cv2img = (cv2img.astype(np.float32) / 127.5) - 1.0

        pixel_hint = get_pixels(cv2img, proportion)

        # cv2.imwrite(str(j) +'-.jpg', (pixel_hint + 1) * 127.5)

        return dict(txt=item['txt'].decode("utf-8"), jpg=cv2img, hint=pixel_hint)

    def get_canny_item(item):
        np_array = np.frombuffer(item['jpg'], np.uint8)
        cv2img = cv2.cvtColor(cv2.imdecode(np_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        cv2img = cv2.resize(cv2img, (side_length, side_length), interpolation=cv2.INTER_AREA) 

        middle = random.randint(30, 225)
        offset = random.randint(20, 125)

        lower = max(15, middle - offset)
        upper = min(240, middle + offset)

        canny_hint = cv2.Canny(cv2img, lower, upper)
        canny_hint = np.repeat(canny_hint, 3, axis=1).reshape(cv2img.shape)

        cv2img = (cv2img.astype(np.float32) / 127.5) - 1.0
        canny_hint = (canny_hint.astype(np.float32) / 127.5) - 1.0
        
        return dict(txt=item['txt'].decode("utf-8"), jpg=cv2img, hint=canny_hint)

    if hint_type == 'pixels':
        return get_pixels_item

    if hint_type == 'canny':
        if proportion != None:
            raise ValueError('do not pass in proportion parameter when training with canny hints')

        return get_canny_item
    
    raise ValueError(f'Unexpected dataset loading method: {hint_type}')

def make_nullifier(proportion):

    def nullifier(item):
        if random.random() > proportion:
            item['txt'] = ''
        return item
    
    return nullifier


def nice_prompts(item):
    item['txt'] = '4k, trending on artstation ' + item['txt'] 
    
    return item

def load_laion(hint_type, batch_size, train_url, test_url, 
    resize, proportion, test_batch_size=None, text_proportion=1.0):
    parser = make_parser(hint_type, proportion, resize)
    train = wds.WebDataset(train_url).map(parser)
    test = wds.WebDataset(test_url).map(parser)

    if text_proportion == 0:
        raise ValueError('text proportion is zero', text_proportion)

    if text_proportion < 1.0:
        text_nullifier = make_nullifier(text_proportion)
        train = train.map(text_nullifier)

    train_dl = DataLoader(train, batch_size=batch_size, num_workers=4)
    # NOTE: num_workers MUST REMAIN 0 for test or we will get DIFFERENT
    # batches each time we log images
    test_dl = DataLoader(test, batch_size=batch_size if test_batch_size is None else test_batch_size, num_workers=0)

    return train_dl, test_dl

# if __name__ == '__main__':
#     trn, tst = load_laion(
#         'canny',
#         4, 
#         "training/laion-100k-data/{00000..00198}.tar",
#         "training/laion-100k-data/00199.tar",
#         resize=512,
#         proportion=None
#     )

#     i = iter(tst)

#     for j in range(4):
#         out = next(i)
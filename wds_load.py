import webdataset as wds
import numpy as np
import cv2
from torch.utils.data import DataLoader
    
def get_pixels(img, proportion):
    mask = np.random.choice([0, 1], size=img.shape[:2], p=[1-proportion, proportion])

    return img * np.repeat(mask, 3, axis=1).reshape(img.shape)

def make_parser(proportion, side_length):

    def get_item(item):
        np_array = np.frombuffer(item['jpg'], np.uint8)
        cv2img = cv2.cvtColor(cv2.imdecode(np_array, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        cv2img = cv2.resize(cv2img, (side_length, side_length), interpolation=cv2.INTER_AREA) 
        # cv2.imwrite(str(j) +'.jpg', cv2img)
        cv2img = (cv2img.astype(np.float32) / 127.5) - 1.0

        pixel_hint = get_pixels(cv2img, proportion)

        # cv2.imwrite(str(j) +'-.jpg', (pixel_hint + 1) * 127.5)

        return dict(txt=item['txt'].decode("utf-8"), jpg=cv2img, hint=pixel_hint)

    return get_item

def load_laion(batch_size, train_url, test_url, resize, proportion):
    parser = make_parser(proportion, resize)
    train = wds.WebDataset(train_url).map(parser)
    test = wds.WebDataset(test_url).map(parser)

    train_dl = DataLoader(train, batch_size=batch_size, num_workers=0)
    test_dl = DataLoader(test, batch_size=batch_size, num_workers=0)

    return train_dl, test_dl

if __name__ == '__main__':
    trn, tst = load_laion(
        4, 
        "training/laion-100k-data/{00000..00198}.tar",
        "training/laion-100k-data/00199.tar",
        resize=512,
        proportion=0.2
    )

    i = iter(tst)

    for j in range(4):
        out = next(i)
        
        # cv2.imwrite(str(j) +'.jpg', out['jpg'])

        # for k, v in out.items():
            # print

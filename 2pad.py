import torch
_ = torch.manual_seed(42)
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.multimodal import CLIPScore
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils.generic import PaddingStrategy


txt = "a photo of a cat a a a a a aa a a a  a a aa  a aa apple i like big butts and i cannot lie, a photo of a cat a a a a a aa a a a  a a aa  a aa apple i like big butts and i cannot lie, a photo of a cat a a a a a aa a a a  a a aa  a aa apple i like big butts and i cannot lie, a photo of a cat a a a a a aa a a a  a a aa  a aa apple i like big butts and i cannot lie"

c = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

# c.processor.tokenizer.set_truncation_and_padding("max_length", TruncationStrategy.ONLY_FIRST, 77, 0, 1)

# print(dir(c.processor.tokenizer))
# print(type(c.processor.tokenizer))

# print('asdfasdfasdff', c.processor.tokenizer.model_max_length)

a = ['Classic Trio 1978 by elfquest', 
 'Integritatea in mediul juridic, educational si privat', 
 '"Lenovo ThinkPad P53 Workstation Laptop (Intel i7-9750H 6-Core, 32GB RAM, 1TB SATA SSD, Quadro T1000, 15.6"" Full HD (1920x1080), Fingerprint, Bluetooth, Webcam, 2xUSB 3.1, 1xHDMI, SD Card, Win 10 Pro)"', 
 'design of german helmet melon lightweight adjustable kids helmet german design', 
 'Poster of Wheelman', 
 '"The Beatles ""Yesterday and Today"" Reproduction ""Butcher Cover Recall Letters"""', 
 'vinyl wallpapers 3d wallppapers 3d wall paper for home decoration', 
 'What is Social Media and Why Should...']

for i in a:
    print(c.processor.tokenizer.tokenize(i, i))

score = clip_score(
[
    torch.randint(255, (3, 224, 224)),
    torch.randint(255, (3, 224, 224)),
    torch.randint(255, (3, 224, 224)),
    torch.randint(255, (3, 224, 224)),
    torch.randint(255, (3, 224, 224)),
    torch.randint(255, (3, 224, 224)),
    torch.randint(255, (3, 224, 224)),
    torch.randint(255, (3, 224, 224)),
],  
a, 
"openai/clip-vit-base-patch16")
print(score.detach())
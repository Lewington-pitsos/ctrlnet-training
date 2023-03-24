from collections import defaultdict
import wandb
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPProcessor, CLIPModel

class ClipScore():
    # default to the 512x512 model, there is also a 768x768 model
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def __call__(self, image_batch, text_batch):
        with torch.no_grad():
            if isinstance(image_batch, torch.Tensor) and image_batch.ndim == 4:
                image_batch = torch.split(image_batch, [1] * image_batch.shape[0])

            inputs = self.processor(
                text=text_batch, 
                images=[torch.squeeze(i) for i in image_batch], 
                return_tensors="pt", padding=True, 
                truncation=True
            )
            outputs = self.model(**inputs)

            text_embeds = outputs['text_embeds']
            image_embeds = outputs['image_embeds']

            scores= (F.cosine_similarity(text_embeds, image_embeds) * 100).clip(0)
        return scores.detach()

def detach_weights(model):
    weights = {
        'early_input': model.control_model.input_blocks[5][0].in_layers[2].weight.detach().cpu(),
        'late_input':  model.control_model.input_blocks[10][0].in_layers[2].weight.detach().cpu(),
        'middle': model.control_model.middle_block[2].in_layers[2].weight.detach().cpu(),
    }

    return weights

def weight_euclidian(weights, model):
    new_weights = detach_weights(model)

    metrics = {}

    for key, weight in weights.items():
        metrics[key + '_euclidian'] = (new_weights[key] - weight).pow(2).sum().sqrt()
    
    return metrics

class ImageLogger(Callback):
    def __init__(self, tst_loader, model, batch_frequency=2000, max_images=8, n_batches=16, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.tst_loader = tst_loader
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.n_batches = n_batches
        self.guidence_scale = 9.0 if log_images_kwargs is None else log_images_kwargs['unconditional_guidance_scale']
        self.output_name = f"samples_cfg_scale_{self.guidence_scale:.2f}"
        self.clip_score = ClipScore()
        self.fid_score = FrechetInceptionDistance(feature=64)
        self.initial_weights = detach_weights(model)

    def to_uint8(self, img):
        return ((img + 1) * 127.5).to(torch.uint8)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and 
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            wandb.log(weight_euclidian(self.initial_weights, pl_module)) 

            dl = iter(self.tst_loader)

            clip_mean = 0
            batch_count = 0
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            img_dict = defaultdict(lambda: [])
            prompts = []

            for i in range(self.n_batches):
                tst_batch =  next(dl)

                with torch.no_grad():
                    images = pl_module.log_images(tst_batch, split=split, **self.log_images_kwargs)

                for k in images:
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()

                clip_mean += self.clip_score(images[self.output_name], tst_batch['txt']).mean()
                samples_in_batch = images[self.output_name].shape[0]
                batch_count += samples_in_batch

                if samples_in_batch > 1:
                    self.fid_score.update(self.to_uint8(images['reconstruction']), real=True)
                    self.fid_score.update(self.to_uint8(images[self.output_name]), real=False)

                if i <= 2:
                    prompts.extend(tst_batch[pl_module.cond_stage_key])
                    for key in images:
                        img_dict[key].append(images[key])

            for key in img_dict:
                img_dict[key] = wandb.Image(torch.cat(img_dict[key]))

            img_dict['conditioning'] = prompts

            wandb.log(img_dict)
            
            wandb.log({
                'clip_score': clip_mean / batch_count,
                'fid_64': self.fid_score.compute(),
            })
            self.fid_score.reset()            

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=None):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

class ZeroConvLogger(Callback):
    def __init__(self, batch_frequency=50) -> None:
        super().__init__()
        self.batch_frequency=batch_frequency

    def check_frequency(self, check_idx):
        return check_idx % self.batch_frequency == 0
    
    def on_train_start(self, trainer, pl_module):
        self.on_train_batch_end(trainer, pl_module, None, None, 0, None)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if self.check_frequency(batch_idx) and pl_module.control_model is not None:
            with torch.no_grad():
                count = 0

                weight_mean = 0
                weight_std = 0
                weight_frobenius_norm = 0

                for i, c in enumerate(pl_module.control_model.zero_convs):  
                    layer = c[0]
                    if hasattr(layer, 'weight'):            

                        weight_mean += layer.weight.mean()
                        weight_std += layer.weight.std()
                        weight_frobenius_norm += torch.norm(layer.weight)

                        wandb.log({
                            f'zc-{i}': {
                                'zc-weight-std': layer.weight.std(),
                                'zc-weight-mean': layer.weight.mean(),
                                'zc-weight-frob': torch.norm(layer.weight),
                            }
                        })
                    count += 1

                if hasattr(c, 'weight_factor'):
                    wandb.log({
                        'weight-factor': c.weight_factor
                    })

                wandb.log({
                    'zc-all-weights-mean': weight_mean / count,
                    'zc-all-weights-frobenius': weight_frobenius_norm / count,
                    'zc-all-weights-std': weight_std / count,
                })

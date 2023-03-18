import wandb
import torch
from pytorch_lightning.callbacks import Callback
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image.fid import FrechetInceptionDistance
# from torchmetrics.functional.multimodal.clip_score import clip_score


class ImageLogger(Callback):
    def __init__(self, tst_loader, batch_frequency=2000, max_images=8, n_batches=16, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, unconditional_guidence_scale=9.0,
                 log_images_kwargs=None):
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
        self.guidence_scale = unconditional_guidence_scale
        self.output_name = f"samples_cfg_scale_{self.guidence_scale:.2f}"
        # self.clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16", truncate=True)
        self.fid_64 = FrechetInceptionDistance(feature=64)
        self.fid_768 = FrechetInceptionDistance(feature=768)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            dl = iter(self.tst_loader)

            logger = type(pl_module.logger)

            mean_clip = 0
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            for i in range(self.n_batches):
                tst_batch = next(dl)

                with torch.no_grad():
                    images = pl_module.log_images(tst_batch, split=split, **self.log_images_kwargs)

                for k in images:
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()

                # CLIP does not truncate sequences that are longer than 77 tokens and instead throws errors
                # so we are adding this very rough "truncate-to-50-words" code
                # text = [" ".join(t.split(" ")[:50]) for t in tst_batch['txt']]
                # print(text)

                # mean_clip += self.clip_metric(images[self.output_name], text).detach()
                
                self.fid_64.update(images['reconstruction'], real=True)
                self.fid_64.update(images[self.output_name], real=False)
                self.fid_768.update(images['reconstruction'], real=True)
                self.fid_768.update(images[self.output_name], real=False)

                if i == 0:
                    for key in images:
                        wandb.log({key: wandb.Image(images[key])})
            
            wandb.log({
                'clip_score': mean_clip / self.n_batches,
                'fid_64': self.fid_64.compute(),
                'fid_768': self.fid_768.compute()

            })


            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
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

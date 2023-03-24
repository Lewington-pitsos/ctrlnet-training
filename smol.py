import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import torch.nn as nn
from cldm.cldm import RisingWeight

class Smol(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.control_model=None

        self.inconv = RisingWeight(torch.nn.Conv2d(3, 3, 1))

        # l1 = 

    def configure_optimizers(self):
        pass

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)    

        self.log_dict(dict(val_loss=loss))

        return loss
    
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def _step(self, batch, batch_idx):
        img = batch['jpg'].permute(0, 3, 1, 2)

        y_hat = self(img, batch['pixel_hint'], 0, None)

        loss = F.cross_entropy(y_hat, img)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)    
    
        self.log_dict(dict(train_loss=loss))
        return loss

    def forward(self, x, hint, timesteps, context, **kwargs):
        return self.inconv(x)



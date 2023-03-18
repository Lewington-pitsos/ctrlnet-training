import json
import numpy as np
import random
import torch

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

import traceback
import warnings
import sys

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback

from uuid import uuid4
from omegaconf import OmegaConf, DictConfig
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.logger import ImageLogger, ZeroConvLogger
from cldm.model import create_model, load_state_dict
from wds_load import load_laion
import argparse

def create_hparam_model(config_path, **kwargs):
    model=create_model(config_path)
    for key, value in kwargs.items():
        model.hparams[key] = value

    model.learning_rate = model.hparams['learning_rate']

    return model

def perform_training_run(args: DictConfig, tune=False):
    print('commencing run:', args.run_name)

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_hparam_model(
        args.model_config_path,
        batch_size=args.batch_size, 
        learning_rate=args.learning_rate,
        train_url=args.train_url, 
        test_url=args.test_url,
        hint_proportion=args.get('hint_proportion'),
        accumulate_grad_batches=args.get('accumulate_grad_batches'),
        max_steps=args.max_steps,
        input_size=args.input_size,
        run_name=args.run_name,
        hint_type=args.hint_type
    ).cpu()
    if args.resume_path != '':
        model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    train_dl, test_dl = load_laion(
        model.hparams['hint_type'],
        model.hparams['batch_size'], 
        model.hparams['train_url'], 
        model.hparams['test_url'],
        model.hparams['input_size'],
        model.hparams['hint_proportion'], 
    )

    img_logger = ImageLogger(test_dl, batch_frequency=args.img_logger_freq, n_batches=16)
    zc_logger = ZeroConvLogger(args.zc_logger_freq)
    model_path = args.run_name + '-' + str(uuid4()) + '.ckpt'
    dirpath = "checkpoints/"
    save_checkpoints = ModelCheckpoint(dirpath=dirpath, filename=model_path)

    training_logger = True
    if args.use_wandb and not tune:
        wandb_logger = WandbLogger(project=args.experiment_name)
        training_logger = [wandb_logger]
        wandb.init(project=args.experiment_name, config=args)
    
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=1,
        precision=32, 
        callbacks=[img_logger, zc_logger, save_checkpoints], 
        logger=training_logger, 
        accumulate_grad_batches=model.hparams['accumulate_grad_batches'],
        log_every_n_steps=args.zc_logger_freq,
        max_steps=model.hparams['max_steps'],
        # val_check_interval=args.val_check_interval,
        auto_lr_find =tune
    )

    if tune:
        print('initial lr', model.learning_rate)
        trainer.tune(model, train_dl, test_dl)
        print(model.learning_rate)
    else: 
        print("model learning rate", model.learning_rate)

        # trainer.validate(model, test_dl)
        # trainer.fit(model, train_dl, test_dl)
        trainer.fit(model, train_dl)


        if args.use_wandb:
            wandb.save(dirpath + model_path)
            wandb.finish()

parser = argparse.ArgumentParser(description='Perform training runs')
parser.add_argument('--run_file', type=str, default='runs.json', help='Path to the file containing run config')
parser.add_argument('--tune', type=bool, default=False, help='run in tune mode')

args = parser.parse_args()

if __name__ == '__main__':
    with open(args.run_file) as f:
        runs = json.load(f)

    print(runs)
    runs = [OmegaConf.create(run) for run in runs['runs']]

    for run in runs:
        perform_training_run(run, args.tune)
        if args.tune:
            print('tuning run complete, exiting...')
            break
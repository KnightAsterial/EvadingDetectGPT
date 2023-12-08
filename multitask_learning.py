"""Implementation of model-agnostic meta-learning for Omniglot."""
import sys
sys.path.append('..')
import argparse
import os

import numpy as np
import torch

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import nn
import torch.nn.functional as F
from torch import autograd
from torch.utils import tensorboard
# from google_drive_downloader import GoogleDriveDownloader as gdd

# import omniglot
# import util
import dataset
import globals

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk, load_dataset, Dataset
import nltk
nltk.download('punkt')

class Multitask:
    def __init__(self, model, ):
        self.model = model
        self.heads = [nn.inear(in_features=768, out_features=32128, bias=False) for edits in range(0,10)]

        self.optimizer = torch.optim.Adam(model.parameters() + )


    def train(self, dataloader_meta_train, dataloader_meta_val, writer):

            print(f'Starting training at iteration {self._start_train_step}.')
            for i_step, task_batch in enumerate(
                    dataloader_meta_train,
                    start=self._start_train_step
            ):
                self._optimizer.zero_grad()

            loss = uter_step(task_batch, True)

            terloss.backward()
              self._optimizer.step()

                # print(torch.cuda.memory_summary())

                
                if i_step % LOG_INTERVAL == 0:
                    print(
                        f'Iteration {i_step}: '
                        f'loss: {outer_loss.item():.3f}, '
                    )
                    writer.add_scalar('loss/train', outer_loss.item(), i_step)


                if i_step % VAL_INTERVAL == 0:
                    losses = []

                    for val_task_batch in dataloader_meta_val:
                        outer_loss = self._outer_step(val_task_batch, False)
                        losses.append(outer_loss.item())

                    loss = np.mean(losses)

                    print(
                        f'Validation: '
                        f'loss: {loss:.3f}, '

                    )
                    writer.add_scalar('loss/val', loss, i_step)


                if i_step % SAVE_INTERVAL == 0:
                    self._save(i_step)
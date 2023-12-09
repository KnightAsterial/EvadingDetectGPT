"""Implementation of model-agnostic meta-learning for Omniglot."""
import sys
sys.path.append('..')
import argparse
import os
import copy

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

SAVE_INTERVAL = 3000 # Previously 100
LOG_INTERVAL = 100 # Previously 10
VAL_INTERVAL = LOG_INTERVAL * 5

DEVICE='cpu'

class Multitask:
    def __init__(self, model, tokenizer, supported_num_edits, lr, log_dir, device):
        self.lr = lr
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.heads = [nn.Linear(in_features=768, out_features=32128, bias=False, device=device) for edits in range(0,supported_num_edits)]
        for i in range(len(self.heads)):
            head = self.heads[i]
            head.weight = nn.Parameter(torch.clone(self.model.lm_head.weight.detach()))
        self.model.lm_head=self.heads[0] # Temporarily set the 0 head as the main head

        parameters = []
        for head in self.heads:
            parameters.extend(head.parameters())
        parameters.extend(self.model.parameters())

        self._optimizer = torch.optim.Adam(parameters, lr=lr)

        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0
        
    
    def step(self, task_batch):
        loss_batch = []
        for i, task in enumerate(task_batch):
            
            ai_support, human_support, _, _, num_edits = task

            
            # tokenize inputs
            ai_support = ["paraphrase: " + sentence + "</s>" for sentence in ai_support]
            ai_support = self.tokenizer(ai_support, return_tensors="pt", padding=True).to(self.device)
            print(ai_support["input_ids"].device)
            print("MODEL embedding", self.model.encoder.embed_tokens.weight.device)
            human_support = self.tokenizer(human_support, return_tensors="pt", padding=True)["input_ids"].to(self.device)
            
            loss = self.model(**ai_support, labels=human_support).loss
            
            loss_batch.append(loss)

        loss = torch.mean(torch.stack(loss_batch))
        
        return loss

    def test_step(self, task_batch):
        
        generated_output = []
        for i, task in enumerate(task_batch):
            
            _, _, ai_query, _, num_edits = task
            
            # tokenize inputs
            ai_query = ["paraphrase: " + sentence + "</s>" for sentence in ai_query]
            ai_query = self.tokenizer(ai_query, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                generated_output = self.tokenizer.batch_decode(self.model.generate(
                                    **ai_query,
                                    max_length=256,
                                    do_sample=True,
                                    top_k=200,
                                    top_p=0.95,
                                    num_return_sequences=1), skip_special_tokens=True,clean_up_tokenization_spaces=True)
                
        return generated_output

    
    def train(self, dataloader_meta_train, dataloader_meta_val, writer):

        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_meta_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
                
            # since batch size is always 1, this is ok
            num_edits = task_batch[0][-1]
            # replace the head in self.model
            self.model.lm_head = self.heads[num_edits]

            loss = self.step(task_batch)

            loss.backward()
            self._optimizer.step()

            # print(torch.cuda.memory_summary())

            if i_step % LOG_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                )
                writer.add_scalar('loss/train', loss.item(), i_step)


            if i_step % VAL_INTERVAL == 0:
                losses = []

                for val_task_batch in dataloader_meta_val:
                    loss = self._step(val_task_batch, False)
                    losses.append(loss.item())

                val_loss = np.mean(losses)

                print(
                    f'Validation: '
                    f'loss: {val_loss:.3f}, '

                )
                writer.add_scalar('loss/val', val_loss, i_step)


            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)


    def test(self, dataloader_test, data_output_dir, num_ai_paragraphs_to_eval=500):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        # TODO: Implement this method with detectpgt score?
        output = {"ai_sample": [], "rephrased_sample": [], "num_edits": [], "human_sample":[]}
        dataset_ai_samples = load_dataset("aadityaubhat/GPT-wiki-intro", split="train[70%:]")
        dataset_ai_samples = dataset_ai_samples.filter(lambda example: len(example['generated_intro']) > 50)
        def strip_and_split(example):
            example["generated"] = " ".join(example["generated_intro"].strip().split())
            example["sentences"] = nltk.sent_tokenize(example["generated"])
            example["human"] = " ".join(example["wiki_intro"].strip().split())
            return example
        dataset_ai_samples = dataset_ai_samples.map(strip_and_split, remove_columns=dataset_ai_samples.column_names)
        dataset_ai_samples = dataset_ai_samples[:num_ai_paragraphs_to_eval]
        # dataset_ai_samples = {"generated": ["asdsd", "asdads", "asdasd"]
        #                       "sentences": ["as", "as", "as"], ["as", "ds", "asd"], ["asd", "ds", "ds"]}

        for task_batch, generated, sentence, human in zip(dataloader_test, dataset_ai_samples['generated'], dataset_ai_samples['sentences'], dataset_ai_samples['human']):
            ai_support, human_support, _, human_query, num_edits = task_batch[0]
            task_batch[0] = (ai_support, human_support, sentence, human_query, num_edits)
            self.model.lm_head = self.heads[num_edits]
            generated_output_sentences = self.test_step(task_batch)
            generated_sample = " ".join(generated_output_sentences)

            output["human_sample"].append(human)
            output["ai_sample"].append(generated)
            output["rephrased_sample"].append(generated_sample)
            output["num_edits"].append(num_edits)

        ds = Dataset.from_dict(output)
        ds.save_to_disk(data_output_dir)


    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self.heads = state['heads']
            self.model.load_state_dict(state['model_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves parameters and optimizer state_dict as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        optimizer_state_dict = self._optimizer.state_dict()
        model_state_dict = self.model.state_dict()
        torch.save(
            dict(heads=self.heads,
                 model_state_dict=model_state_dict,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):

    print(args)

    if args.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # on MPS the derivative for aten::linear_backward is not implemented ... Waiting for PyTorch 2.1.0
        # DEVICE = "mps"

        # Due to the above, default for now to cpu
        DEVICE = "cpu"
    elif args.device == "gpu" and torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    print("Using device: ", DEVICE)

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/multitask/evadegpt.support_{args.num_support}.query_{args.num_query}.outer_lr_{args.outer_lr}.batch_size_{args.batch_size}.iters_{args.num_train_iterations}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)
    
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws", torch_dtype=torch.bfloat16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")

    multitask = Multitask(
        model,
        tokenizer,
        # args.num_way,
        args.max_num_edits,
        args.outer_lr,
        log_dir,
        DEVICE
    )

    if args.checkpoint_step > -1:
        multitask.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')
        
    # train_dataloader, test_dataloader = dataset.get_pair_dataloader("train[:10%]", 4, 1, 1)

    dataloader_meta_train, dataloader_meta_val, dataloader_test = dataset.get_pair_dataloaders(args.batch_size, args.num_support, args.num_query, args.num_workers, args.num_train_iterations, max_num_edits=args.max_num_edits)

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on {num_training_tasks} tasks with composition: '
            # f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        
        
        multitask.train(
            dataloader_meta_train,
            dataloader_meta_val,
            writer,
        )
    else:
        print(
            f'Testing on tasks with composition '
            # f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}, '
        )

        assert args.batch_size == 1
        assert args.test_output_dir != None
        # dataloader_test = omniglot.get_omniglot_dataloader(
        #     'test',
        #     1,
        #     args.num_way,
        #     args.num_support,
        #     args.num_query,
        #     NUM_TEST_TASKS,
        #     args.num_workers
        # )
        multitask.test(dataloader_test, args.test_output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_support', type=int, default=10,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=1,
                        help='number of query examples per class in a task')
    parser.add_argument('--max_num_edits', type=int, default=10,
                        help='create heads for edits in range (0,max_num_edits)')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=15000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--test_output_dir', type=str, default=None,
                        help='directory that test stores generated outputs to')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--num_workers', type=int, default=2, 
                        help=('needed to specify omniglot dataloader'))
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    # if args.cache == True:
    #     # Download Omniglot Dataset
    #     if not os.path.isdir("./omniglot_resized"):
    #         gdd.download_file_from_google_drive(
    #             file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
    #             dest_path="./omniglot_resized.zip",
    #             unzip=True,
    #         )
    #     assert os.path.isdir("./omniglot_resized")
    # else:
    main(args)

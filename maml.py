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
from google_drive_downloader import GoogleDriveDownloader as gdd

import omniglot
import util

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 32
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600

class LoRALayerWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = base_module

        ###
        ### Set up your LoRA-augmented layer here.
        ### You should initialize your parameters so that the residual matrix AB^T is zero,
        ###     but be careful how you do this (i.e., make sure you eventually get
        ###     non-zero gradients to both matrices during fine-tuning)!
        ### For randomly initializing the parameters, use torch.randn.
        ### Note: you should use nn.Parameter to wrap your parameters so that they are registered as
        ### learnable.
        ### Initialization hint: what do the gradients look like after 1 and 2 steps of fine-tuning
        ###     if you initialize both A and B to zero? What about if just one is zero?
        ###
        self.lora_A, self.lora_B = None, None
        # random initialize lora_A
        # intialize lora_B to zero, gradient for lora_A will become non zero second round
        # dim of AB^T = base_module.weight.size
        self.lora_A = nn.Parameter(torch.randn((base_module.weight.size(0), lora_rank)))
        self.lora_B = nn.Parameter(torch.zeros((base_module.weight.size(1), lora_rank)))
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_module(x)  # The output of the pre-trained module.
        ### Perform the forward pass of your LoRA-augmented layer here.
        ### Note: you don't need to ever explicitly construct the matrix AB^T.
        ### Hint: matrix multiplication is associative.
        ###

        ## YOUR CODE HERE, complete for Q2.2b
        # X(W + AB^T) = XW + XAB^T = XW + (XA)B^T
        return base_out + (x @ self.lora_A) @ self.lora_B.T
    
def get_lora_model(rank):
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    for m in model.encoder.block:
        m.layer[0].SelfAttention.q = LoRALayerWrapper(m.layer[0].SelfAttention.q, rank)
        m.layer[0].SelfAttention.k = LoRALayerWrapper(m.layer[0].SelfAttention.k, rank)
        m.layer[0].SelfAttention.v = LoRALayerWrapper(m.layer[0].SelfAttention.v, rank)
        m.layer[0].SelfAttention.o = LoRALayerWrapper(m.layer[0].SelfAttention.o, rank)

        m.layer[1].DenseReluDense.wi = LoRALayerWrapper(m.layer[1].DenseReluDense.wi, rank)
        m.layer[1].DenseReluDense.wo = LoRALayerWrapper(m.layer[1].DenseReluDense.wo, rank)

    for m in model.decoder.block:
        m.layer[0].SelfAttention.q = LoRALayerWrapper(m.layer[0].SelfAttention.q, rank)
        m.layer[0].SelfAttention.k = LoRALayerWrapper(m.layer[0].SelfAttention.k, rank)
        m.layer[0].SelfAttention.v = LoRALayerWrapper(m.layer[0].SelfAttention.v, rank)
        m.layer[0].SelfAttention.o = LoRALayerWrapper(m.layer[0].SelfAttention.o, rank)

        m.layer[1].EncDecAttention.q = LoRALayerWrapper(m.layer[1].EncDecAttention.q, rank)
        m.layer[1].EncDecAttention.k = LoRALayerWrapper(m.layer[1].EncDecAttention.k, rank)
        m.layer[1].EncDecAttention.v = LoRALayerWrapper(m.layer[1].EncDecAttention.v, rank)
        m.layer[1].EncDecAttention.o = LoRALayerWrapper(m.layer[1].EncDecAttention.o, rank)


        m.layer[2].DenseReluDense.wi = LoRALayerWrapper(m.layer[2].DenseReluDense.wi, rank)
        m.layer[2].DenseReluDense.wo = LoRALayerWrapper(m.layer[2].DenseReluDense.wo, rank)


class MAML:
    """Trains and assesses a MAML."""

    def __init__(
            self,
            # num_outputs,
            model,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            log_dir,
            device
    ):
        """Inits MAML.

        The network consists of four convolutional blocks followed by a linear
        head layer. Each convolutional block comprises a convolution layer, a
        batch normalization layer, and ReLU activation.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.

        Args:
            num_outputs (int): dimensionality of output, i.e. number of classes
                in a task
            num_inner_steps (int): number of inner-loop optimization steps
            inner_lr (float): learning rate for inner-loop optimization
                If learn_inner_lrs=True, inner_lr serves as the initialization
                of the learning rates.
            learn_inner_lrs (bool): whether to learn the above
            outer_lr (float): learning rate for outer-loop optimization
            log_dir (str): path to logging directory
            device (str): device to be used
        """
        meta_parameters = {}

        self.device = device

        # construct feature extractor
        # in_channels = NUM_INPUT_CHANNELS

        self.model = model
        for i, m in enumerate(model.modules()):
            if isinstance(m, LoRALayerWrapper):
                meta_parameters[f"{i}_A"] = m.lora_A
                meta_parameters[f"{i}_B"] = m.lora_B
                



        # for i in range(NUM_CONV_LAYERS):
        #     meta_parameters[f'conv{i}'] = nn.init.xavier_uniform_(
        #         torch.empty(
        #             NUM_HIDDEN_CHANNELS,
        #             in_channels,
        #             KERNEL_SIZE,
        #             KERNEL_SIZE,
        #             requires_grad=True,
        #             device=self.device
        #         )
        #     )
        #     meta_parameters[f'b{i}'] = nn.init.(
        #         torch.empty(zeros_
        #             NUM_HIDDEN_CHANNELS,
        #             requires_grad=True,
        #             device=self.device
        #         )
        #     )
        #     in_channels = NUM_HIDDEN_CHANNELS

        # construct linear head layer
        # meta_parameters[f'w{NUM_CONV_LAYERS}'] = nn.init.xavier_uniform_(
        #     torch.empty(
        #         num_outputs,
        #         NUM_HIDDEN_CHANNELS,
        #         requires_grad=True,
        #         device=self.device
        #     )
        # )
        # meta_parameters[f'b{NUM_CONV_LAYERS}'] = nn.init.zeros_(
        #     torch.empty(
        #         num_outputs,
        #         requires_grad=True,
        #         device=self.device
        #     )
        # )

        self._meta_parameters = meta_parameters
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._outer_lr = outer_lr

        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )

        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0


    '''
    IDEA

    init:
        model = gpt2
        meta_parameters = []
        for layer in gpt2:
            layer = LoraWrapperLayer(layer)
            meta_parameters.extend [layer.lora_A, layer.lora_B] )
        optimizer = adam(meta_parameters)

    __inner_loop:
        parameter_copy = torch.clone(meta_parameters)

        for layer in gpt2:
            layer.loraA = parameter_copy[layer][A]         <<-- different from hw2 bc we substitute in the copies into the original model
            layer.loraB = parameter_copy[layer][B]          <<-- instead of copying all the parameters and appying them manually

        for iter in inner loop
            result = model(inputs)                          <<-- different from hw2 bc we call the model with substituted parameters instead of copying manually
            calculate loss                                     
            torch.autograd.grad(loss, parameter_copy)
            manually apply gradients on parameter_copy

        return loss

    __outer_loop
        run __inner_loop

        for layer in gpt2:
            layer.loraA = original_meta_parameters[layer][A]         <<-- different from hw2 bc we substitute in the copies into the original model
            layer.loraB = original_meta_parameters[layer][B]          <<-- instead of copying all the parameters and appying them manually

        if test: return results of inner_loop

        if train:
            loss.backward()
            optimzer.step()

    '''

    def _forward(self, input, labels, model):
        """Computes predicted classification logits.

        Args:
            inputs (Dict): dict of tokenized inputs
                input_ids:
                attention_mask:

        Returns:
            the cross-entropy loss between the input and the label
        """

        # https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5ForConditionalGeneration

        

    def _inner_loop(self, images, labels, train, model, tokenizer):
        """Computes the adapted network parameters via the MAML inner loop.\
        
        input is 1 task (k human/ai pairs)
        NOTE: WILL MODIFY ORIGINAL MODEL IN PLACE TO HAVE NEW PHI PARAMETERS

        Args:
            input (Dict)
                input_ids (Tensor): has dimensions (k x max_length)
                attention_mask (Tensor): has dimensions (k x maxlength)
            labels (Tensor): has dimension (k x ou)putlength

        Returns:
            parameters (dict[str, Tensor]): adapted network parameters    
        """
        copied_parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }

        # put the copied parameters into our model
        for i, m in enumerate(model.modules()):
            if isinstance(m, LoRALayerWrapper):
                m.lora_A = copied_parameters[f"{i}_A"]
                m.lora_B = copied_parameters[f"{i}_B"]
        
        

        # set lora weights        
        for _ in range(self._num_inner_steps):
            
            # tokenize inputs
            loss = model(**input, labels=labels).loss.item()

            gradients = autograd.grad(loss, copied_parameters.values(), create_graph=train)
            for i, k in enumerate(copied_parameters.keys()):
                copied_parameters[k] = copied_parameters[k] - self._inner_lrs[k] * gradients[i]
        
        ### END CODE HERE ###
        return copied_parameters

    def _outer_step(self, task_batch, train):
        """Computes the MAML loss and metrics on a batch of tasks.

        Args:
            task_batch (tuple): batch of tasks from an Omniglot DataLoader
            train (bool): whether we are training or evaluating

        Returns:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
            accuracies_support (ndarray): support set accuracy over the
                course of the inner loop, averaged over the task batch
                shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted
                parameters, averaged over the task batch
        """
        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            
            ai_support, human_support, ai_query, human_query = task
            
            ai_support = ai_support.to(self.device)
            human_support = human_support.to(self.device)
            ai_query = ai_query.to(self.device)
            human_query = human_query.to(self.device)
            
            parameters = self._inner_loop(ai_support, human_support, train)
            # Use F.cross_entropy to compute classification losses.
            loss = self._forward(ai_query, parameters, )
            outer_loss_batch.append(loss)
            # Use util.score to compute accuracies.
            # Make sure to populate outer_loss_batch, accuracies_support_batch,
            # and accuracy_query_batch.
            # support accuracy: The first element (index 0) should be the accuracy before any steps are taken.
            accuracy_query_batch.append(util.score(logits, labels_query))
        
            ### END CODE HERE ###
        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        accuracies_support = np.mean(
            accuracies_support_batch,
            axis=0
        )
        accuracy_query = np.mean(accuracy_query_batch)
        return outer_loss, accuracies_support, accuracy_query

    def train(self, dataloader_meta_train, dataloader_meta_val, writer):
        """Train the MAML.

        Consumes dataloader_meta_train to optimize MAML meta-parameters
        while periodically validating on dataloader_meta_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_meta_train (DataLoader): loader for train tasks
            dataloader_meta_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_meta_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            outer_loss, accuracies_support, accuracy_query = (
                self._outer_step(task_batch, train=True)
            )
            outer_loss.backward()
            self._optimizer.step()

            if i_step % LOG_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {outer_loss.item():.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracies_support[0]:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracies_support[-1]:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_query:.3f}'
                )
                writer.add_scalar('loss/train', outer_loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/pre_adapt_support',
                    accuracies_support[0],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_support',
                    accuracies_support[-1],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_query',
                    accuracy_query,
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                losses = []
                accuracies_pre_adapt_support = []
                accuracies_post_adapt_support = []
                accuracies_post_adapt_query = []
                for val_task_batch in dataloader_meta_val:
                    outer_loss, accuracies_support, accuracy_query = (
                        self._outer_step(val_task_batch, train=False)
                    )
                    losses.append(outer_loss.item())
                    accuracies_pre_adapt_support.append(accuracies_support[0])
                    accuracies_post_adapt_support.append(accuracies_support[-1])
                    accuracies_post_adapt_query.append(accuracy_query)
                loss = np.mean(losses)
                accuracy_pre_adapt_support = np.mean(
                    accuracies_pre_adapt_support
                )
                accuracy_post_adapt_support = np.mean(
                    accuracies_post_adapt_support
                )
                accuracy_post_adapt_query = np.mean(
                    accuracies_post_adapt_query
                )
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracy_pre_adapt_support:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracy_post_adapt_support:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_post_adapt_query:.3f}'
                )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/pre_adapt_support',
                    accuracy_pre_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_support',
                    accuracy_post_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_query',
                    accuracy_post_adapt_query,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

    def test(self, dataloader_test):
        """Evaluate the MAML on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            _, _, accuracy_query = self._outer_step(task_batch, train=False)
            accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

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
            self._meta_parameters = state['meta_parameters']
            self._inner_lrs = state['inner_lrs']
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
        torch.save(
            dict(meta_parameters=self._meta_parameters,
                 inner_lrs=self._inner_lrs,
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
        log_dir = f'./logs/maml/omniglot.way_{args.num_way}.support_{args.num_support}.query_{args.num_query}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.batch_size}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    maml = MAML(
        args.num_way,
        args.num_inner_steps,
        args.inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        log_dir,
        DEVICE
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on {num_training_tasks} tasks with composition: '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_meta_train = omniglot.get_omniglot_dataloader(
            'train',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            num_training_tasks,
            args.num_workers
        )
        dataloader_meta_val = omniglot.get_omniglot_dataloader(
            'val',
            args.batch_size,
            args.num_way,
            args.num_support,
            args.num_query,
            args.batch_size * 4,
            args.num_workers
        )
        maml.train(
            dataloader_meta_train,
            dataloader_meta_val,
            writer
        )
    else:
        print(
            f'Testing on tasks with composition '
            f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        dataloader_test = omniglot.get_omniglot_dataloader(
            'test',
            1,
            args.num_way,
            args.num_support,
            args.num_query,
            NUM_TEST_TASKS,
            args.num_workers
        )
        maml.test(dataloader_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_way', type=int, default=5,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=1,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.4,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=15000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--num_workers', type=int, default=2, 
                        help=('needed to specify omniglot dataloader'))
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()

    if args.cache == True:
        # Download Omniglot Dataset
        if not os.path.isdir("./omniglot_resized"):
            gdd.download_file_from_google_drive(
                file_id="1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI",
                dest_path="./omniglot_resized.zip",
                unzip=True,
            )
        assert os.path.isdir("./omniglot_resized")
    else:
        main(args)



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

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 32
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 3000 # Previously 100
LOG_INTERVAL = 100 # Previously 10
VAL_INTERVAL = LOG_INTERVAL * 5
NUM_TEST_TASKS = 600

class LoRALayerWrapper(nn.Module):
    def __init__(self, base_module: nn.Module, lora_rank: int, device):
        super().__init__()

        self.base_module = base_module
        self.weight = base_module.weight # to satisfy "isinstance(self.wo.weight, torch.Tensor)" check for T5.DenseReluDense layer
                                         # https://github.com/huggingface/transformers/blob/7ee995fd9c692761c4601ddbffa2ac2ec9f27b0b/src/transformers/models/t5/modeling_t5.py#L292C5-L292C5

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
        self.lora_A = torch.randn((base_module.weight.size(0), lora_rank), requires_grad=True, device=device, dtype=torch.bfloat16)
        self.lora_B = torch.zeros((base_module.weight.size(1), lora_rank), requires_grad=True, device=device, dtype=torch.bfloat16)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_module(x)  # The output of the pre-trained module.
        ### Perform the forward pass of your LoRA-augmented layer here.
        ### Note: you don't need to ever explicitly construct the matrix AB^T.
        ### Hint: matrix multiplication is associative.
        ###

        ## YOUR CODE HERE, complete for Q2.2b
        # For nn.Linear layer: X(W + AB^T)^T = XW^T + XBA^T = XW + (XB)A^T
        return base_out + (x @ self.lora_B) @ self.lora_A.T
    
def get_lora_model(rank, device):
    # tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws", torch_dtype=torch.bfloat16).to(device)
    for m in model.encoder.block:
        m.layer[0].SelfAttention.q = LoRALayerWrapper(m.layer[0].SelfAttention.q, rank, device)
        m.layer[0].SelfAttention.k = LoRALayerWrapper(m.layer[0].SelfAttention.k, rank, device)
        m.layer[0].SelfAttention.v = LoRALayerWrapper(m.layer[0].SelfAttention.v, rank, device)
        m.layer[0].SelfAttention.o = LoRALayerWrapper(m.layer[0].SelfAttention.o, rank, device)

        m.layer[1].DenseReluDense.wi = LoRALayerWrapper(m.layer[1].DenseReluDense.wi, rank, device)
        m.layer[1].DenseReluDense.wo = LoRALayerWrapper(m.layer[1].DenseReluDense.wo, rank, device)

    for m in model.decoder.block:
        m.layer[0].SelfAttention.q = LoRALayerWrapper(m.layer[0].SelfAttention.q, rank, device)
        m.layer[0].SelfAttention.k = LoRALayerWrapper(m.layer[0].SelfAttention.k, rank, device)
        m.layer[0].SelfAttention.v = LoRALayerWrapper(m.layer[0].SelfAttention.v, rank, device)
        m.layer[0].SelfAttention.o = LoRALayerWrapper(m.layer[0].SelfAttention.o, rank, device)

        m.layer[1].EncDecAttention.q = LoRALayerWrapper(m.layer[1].EncDecAttention.q, rank, device)
        m.layer[1].EncDecAttention.k = LoRALayerWrapper(m.layer[1].EncDecAttention.k, rank, device)
        m.layer[1].EncDecAttention.v = LoRALayerWrapper(m.layer[1].EncDecAttention.v, rank, device)
        m.layer[1].EncDecAttention.o = LoRALayerWrapper(m.layer[1].EncDecAttention.o, rank, device)


        m.layer[2].DenseReluDense.wi = LoRALayerWrapper(m.layer[2].DenseReluDense.wi, rank, device)
        m.layer[2].DenseReluDense.wo = LoRALayerWrapper(m.layer[2].DenseReluDense.wo, rank, device)
    
    for param in model.parameters():
        param.requires_grad = False
    return model


class MAML:
    """Trains and assesses a MAML."""

    def __init__(
            self,
            # num_outputs,
            model,
            tokenizer,
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
        self.tokenizer = tokenizer
        for i, m in enumerate(model.modules()):
            if isinstance(m, LoRALayerWrapper):
                meta_parameters[f"{i}_A"] = m.lora_A
                meta_parameters[f"{i}_B"] = m.lora_B

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


    def _inner_loop(self, ai_text, human_text, train):
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
        # print("\/ \/ \/ started inner loop")
        # print(torch.cuda.memory_summary())
        copied_parameters = {
            k: torch.clone(v)
            for k, v in self._meta_parameters.items()
        }

        # put the copied parameters into our model
        for i, m in enumerate(self.model.modules()):
            if isinstance(m, LoRALayerWrapper):
                m.lora_A = copied_parameters[f"{i}_A"]
                m.lora_B = copied_parameters[f"{i}_B"]

        # print("after copied param")
        # print(torch.cuda.memory_summary())
        

        # set lora weights        
        for _ in range(self._num_inner_steps):
            
            # TODO: not sure if this is working
            loss = self.model(**ai_text, labels=human_text).loss

            gradients = autograd.grad(loss, copied_parameters.values(), create_graph=train)
            # print("after autograd")
            # print(torch.cuda.memory_summary())

            for i, k in enumerate(copied_parameters.keys()):
                copied_parameters[k] = copied_parameters[k] - self._inner_lrs[k] * gradients[i]

        # print("/\ /\ /\ ended inner loop")
        ### END CODE HERE ###
        return copied_parameters
    

    def _outer_step(self, task_batch, train):
        outer_loss_batch = []
        for i, task in enumerate(task_batch):
            
            _, _, ai_query, human_query, num_edits = task
            
            # tokenize inputs
            ai_query = ["paraphrase: " + sentence + "</s>" for sentence in ai_query]
            ai_query = self.tokenizer(ai_query, return_tensors="pt", padding=True).to(self.device)
            human_query = self.tokenizer(human_query, return_tensors="pt", padding=True)["input_ids"].to(self.device)


            head = self.heads[num_edits]
            model.lm_head =
            # Model still has the PHI parameters set inside self._inner_loop
            loss = self.model(**ai_query, labels=human_query).loss
            
            outer_loss_batch.append(loss)

            # Use util.score to compute accuracies.
            # Make sure to populate outer_loss_batch, accuracies_support_batch,
            # and accuracy_query_batch.
            # support accuracy: The first element (index 0) should be the accuracy before any steps are taken.
            # accuracy_query_batch.append(util.score(logits, labels_query))
        
            ### END CODE HERE ###
        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        return outer_loss
    

    def _outer_step_test(self, task_batch, train, skip_innerloop=False):
        
        outer_loss_batch = []
        generated_output = []
        for i, task in enumerate(task_batch):
            
            ai_support, human_support, ai_query, human_query, num_edits = task
            
            # tokenize inputs
            ai_support = ["paraphrase: " + sentence + "</s>" for sentence in ai_support]
            ai_query = ["paraphrase: " + sentence + "</s>" for sentence in ai_query]
            ai_support = self.tokenizer(ai_support, return_tensors="pt", padding=True).to(self.device)
            human_support = self.tokenizer(human_support, return_tensors="pt", padding=True)["input_ids"].to(self.device)
            ai_query = self.tokenizer(ai_query, return_tensors="pt", padding=True).to(self.device)
            human_query = self.tokenizer(human_query, return_tensors="pt", padding=True)["input_ids"].to(self.device)
            
            if (not skip_innerloop):
                parameters = self._inner_loop(ai_support, human_support, train)

            # Model still has the PHI parameters set inside self._inner_loop
            if train:
                loss = self.model(**ai_query, labels=human_query).loss
                outer_loss_batch.append(loss)
            else:
                with torch.no_grad():
                    generated_output = self.tokenizer.batch_decode(self.model.generate(
                                        **ai_query,
                                        max_length=256,
                                        do_sample=True,
                                        top_k=200,
                                        top_p=0.95,
                                        num_return_sequences=1), skip_special_tokens=True,clean_up_tokenization_spaces=True)
                    
            

            # Use util.score to compute accuracies.
            # Make sure to populate outer_loss_batch, accuracies_support_batch,
            # and accuracy_query_batch.
            # support accuracy: The first element (index 0) should be the accuracy before any steps are taken.
            # accuracy_query_batch.append(util.score(logits, labels_query))
        
            ### END CODE HERE ###
        
        if train:
            outer_loss = torch.mean(torch.stack(outer_loss_batch))
            return outer_loss
        else:
            return generated_output
    
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

            outer_loss = self._outer_step(task_batch, True)

            outer_loss.backward()
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

    def test(self, dataloader_test, data_output_dir, num_ai_paragraphs_to_eval=500, skip_innerloop=False):
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
            generated_output_sentences = self._outer_step_test(task_batch, train=False, skip_innerloop=skip_innerloop)
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
        log_dir = f'./logs/maml/evadegpt.support_{args.num_support}.query_{args.num_query}.inner_steps_{args.num_inner_steps}.inner_lr_{args.inner_lr}.learn_inner_lrs_{args.learn_inner_lrs}.outer_lr_{args.outer_lr}.batch_size_{args.batch_size}.iters_{args.num_train_iterations}'  # pylint: disable=line-too-long
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)
    
    # Initialize lora model
    model = get_lora_model(8, DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")

    maml = MAML(
        model,
        tokenizer,
        # args.num_way,
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
        
    # train_dataloader, test_dataloader = dataset.get_pair_dataloader("train[:10%]", 4, 1, 1)

    dataloader_meta_train, dataloader_meta_val, dataloader_test = dataset.get_pair_dataloaders(args.batch_size, args.num_support, args.num_query, args.num_workers, args.num_train_iterations)

    if not args.test:
        num_training_tasks = args.batch_size * (args.num_train_iterations -
                                                args.checkpoint_step - 1)
        print(
            f'Training on {num_training_tasks} tasks with composition: '
            # f'num_way={args.num_way}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        
        
        maml.train(
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
            f'test_skip_innerloop={args.test_skip_innerloop}'
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
        maml.test(dataloader_test, args.test_output_dir, skip_innerloop=args.test_skip_innerloop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_support', type=int, default=10,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=1,
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
    parser.add_argument('--test_output_dir', type=str, default=None,
                        help='directory that test stores generated outputs to')
    parser.add_argument('--test_skip_innerloop', default=False, action='store_true',
                        help='should we run the inner loop within test when generating outputs')
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



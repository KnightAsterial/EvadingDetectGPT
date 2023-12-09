"""Dataloading for Omniglot."""

import torch
from torch.utils.data import dataset, sampler, dataloader
from datasets import load_dataset, load_from_disk
import random
import numpy as np
from collections import Counter

NUM_TRAIN_CLASSES = 1100
NUM_VAL_CLASSES = 100
NUM_TEST_CLASSES = 423
NUM_SAMPLES_PER_CLASS = 20

def edit_distance(sent1, sent2):
    sent1_split = sent1.lower().split(' ')
    sent2_split = sent2.lower().split(' ')

    m, n = len(sent1_split), len(sent2_split)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize the DP table
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if sent1_split[i - 1] == sent2_split[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i][j - 1] + 1, dp[i - 1][j] + 1)

    return dp[m][n]


class PairedDataset(dataset.Dataset):
    """Paired dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    def __init__(self, num_support, num_query, dataset):
        """Inits human/text pair dataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()

        # load dataset from huggingface
        self.dataset = dataset

        self.avail_edits = []
        
        def calc_edit_distance(example):
            example['edit_distance'] = edit_distance(example['human_sents'], example['ai_sents'])
            example['ratio'] = example['edit_distance'] / (len(example['ai_sents']) if len(example['ai_sents']) != 0 else float('inf'))
            return example
        
        # TODO: Remove hard coded dataset length
        self.dataset = self.dataset.filter(lambda example: len(example['human_sents']) + len(example['ai_sents']) < 3000).map(calc_edit_distance, num_proc=10)
        print(self.dataset[0])
        print(edit_distance(self.dataset[0]['human_sents'], self.dataset[0]['ai_sents']))
        counter = Counter(self.dataset["edit_distance"])
        self.avail_edits = [k for k, v in counter.items() if v >= num_support+num_query]
        # self.avail_edits = list(set(self.dataset["edit_distance"]))
        # print("OUTPUTTING EXAMPLE WITH", np.sort(np.array(self.dataset["edit_distance"]) / np.array(self.dataset["generated_intro_len"])), "edits!!!")
        print(self.avail_edits)
        # check problem arguments
        assert num_support + num_query <= NUM_SAMPLES_PER_CLASS
        self._num_support = num_support
        self._num_query = num_query

    def __getitem__(self, num_edits):
        """Constructs a task.

        Data for each class is sampled uniformly at random without replacement.

        Args:
            num_edits (in): cnumber of edits between human and AI text
        Returns:
            ai_support (List[String]): task support phrases
            human_support (List[String]): task support phrases
            ai_query (List[String]): task query phrases
            human_query (List[String]): task query phrases
            num_edits (int): number of edits between each human and AI text pair
        """
        ai_support, ai_query = [], []
        human_support, human_query = [], []

        # # get a class's examples and sample from them
        # all_file_paths = glob.glob(
        #     os.path.join(self._character_folders[class_idx], '*.png')
        # )
        
        # select (self._num_support + self._num_query) examples from the dataset
        
        pairs = self.dataset.filter(lambda example: example['edit_distance'] == num_edits)
        idxs = np.random.choice(len(pairs), size=self._num_support + self._num_query)
        pairs = pairs.select(idxs)

        ai_support.extend(pairs["human_sents"][:self._num_support])
        ai_query.extend(pairs["human_sents"][self._num_support:])
        human_support.extend(pairs["ai_sents"][:self._num_support])
        human_query.extend(pairs["ai_sents"][self._num_support:])

        
        # # aggregate into tensors
        # ai_support = torch.stack(ai_support)  # shape (N*S, C, H, W)
        # human_support = torch.tensor(human_support)  # shape (N*S)
        # ai_query = torch.stack(ai_query)
        # human_query = torch.tensor(human_query)

        return ai_support, human_support, ai_query, human_query, num_edits


class PairSampler(sampler.Sampler):
    """Samples task specification keys for an OmniglotDataset."""

    def __init__(self, avail_edits, num_tasks_per_epoch, max_num_edits=None):
        """Inits OmniglotSampler.

        Args:
            avail_edits (List[int]: list of edit counts )
        """
        super().__init__(None)
        if max_num_edits==None:
            self._avail_edits = avail_edits
        else:
            filtered_edits = []
            for i in range(max_num_edits):
                if i in avail_edits:
                    filtered_edits.append(i)
            self._avail_edits = filtered_edits
        self.num_tasks_per_epoch = num_tasks_per_epoch

    def __iter__(self):
        return (random.choice(self._avail_edits) for _ in range(self.num_tasks_per_epoch))
    
    def __len__(self):
        return self.clnum_tasks_per_epoch


def identity(x):
    return x


def get_pair_dataloaders(
        batch_size,
        num_support,
        num_query,
        num_workers=2,
        num_tasks_per_epoch=3000,
        max_num_edits=None
):
    """Returns three dataloader.DataLoader for paired human/AI text

    Args:
        batch_size (int): number of tasks per batch
        num_support (int): number of support examples per task
        num_query (int): number of query examples per task
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
    """

    dataset = load_from_disk('./data_t5_wikidoc')
    dataset = dataset.with_format(None)

    # dataset = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    # splits = dataset.train_test_split(test_size=0.1)
    # train = splits["train"]
    # test = splits["test"]
    
    # 90% train, 10% test + validation
    train_testvalid = dataset.train_test_split(test_size=0.1)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    # train_test_valid_dataset = DatasetDict({
    #     'train': train_testvalid['train'],
    #     'test': test_valid['test'],
    #     'valid': test_valid['train']})

    # train_dataset = PairedDataset(num_support, num_query, train)
    # test_dataset = PairedDataset(num_support, num_query, test)
    
    # return [PairedDataset(num_support, num_query, dataset) in [train_testvalid["train"], test_valid["train"], test_valid["test"]]]

    paired_datasets = [PairedDataset(num_support, num_query, split_dataset) for split_dataset in [train_testvalid["train"], test_valid["train"], test_valid["test"]]]
    print(paired_datasets[1].avail_edits)
    return [dataloader.DataLoader(
                dataset=paired_dataset,
                batch_size=batch_size,
                sampler=PairSampler(paired_dataset.avail_edits, num_tasks, max_num_edits=max_num_edits),
                # num_workers=num_workers,
                collate_fn=identity,
                pin_memory=torch.cuda.is_available(),
                drop_last=True) 
            for num_tasks, paired_dataset in zip([num_tasks_per_epoch, 4, 1_000_000], paired_datasets)] #TODO: HARDCODED

    #     dataloader.DataLoader(
    #         dataset=test_dataset,
    #         batch_size=batch_size,
    #         sampler=PairSampler(dataset.avail_edits, num_tasks_per_epoch),
    #         # num_workers=num_workers,
    #         collate_fn=identity,
    #         pin_memory=torch.cuda.is_available(),
    #         drop_last=True
    #     )
    # )

# train_dataloader, test_dataloader = get_pair_dataloader("train[:10%]", 4, 1, 1)
# print(dataload, test_dataloaderer)
# for i, batch in enumerate(dataloader):
#     print(i)
#     # for sample in batch:
#     #     print(sample)

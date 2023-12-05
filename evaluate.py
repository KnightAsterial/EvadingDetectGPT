from customDetectGPT import get_score 
from datasets import load_from_disk, load_dataset
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import argparse
import pickle
import os

def strip_whitespace(example, ai_label, human_label):
    example[human_label] = " ".join(example[human_label].strip().split())
    example[ai_label] = " ".join(example[ai_label].strip().split())

# plot histogram of diff and save fig
# plt.hist(diff, bins=50)
# plt.savefig("histogram.png")

def plot_overlaying_histograms(human_scores, ai_scores, method, human_label, ai_label):
    plt.clf()
    plt.hist(human_scores, bins=50, alpha=0.5, label=human_label)
    plt.hist(ai_scores, bins=50, alpha=0.5, label=ai_label)
    plt.legend(loc='upper right')
    plt.savefig(f"{method}_histogram.png")

def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--method', type=str, default=None, required=False,
                        help='name for this evaluation')
    parser.add_argument('--ai_label', type=str, default=None, required=True,
                        help='ai_label')
    parser.add_argument('--human_label', type=str, default=None, required=True,
                        help='human_label')
    parser.add_argument('--dataset_dir', type=str, default=None, required=True,
                        help='local directory to load huggingface dataset from')
    args = parser.parse_args()
    
    # # CONFIG
    method = args.method
    ai_label = args.ai_label
    human_label = args.human_label
    dataset = load_from_disk(args.dataset_dir)
    
    # method = "dataset_stats"
    # ai_label = "generated"
    # human_label = "wiki_intro"
    # dataset = load_from_disk("./data_t5_wikidoc_para")

    # load if f"{ai_label}_scores.pkl" is present
    if os.path.exists(f"{args.dataset_dir}/{ai_label}_scores.pkl"):
        print(f"Loading {ai_label} from pickle")
        with open(f"{args.dataset_dir}/{ai_label}_scores.pkl", "rb") as f:
            ai_scores = pickle.load(f)
    else:
        print(f"Calculating {ai_label} since it is not saved from before")
        dataset = dataset.map(lambda example: strip_whitespace(example, ai_label, human_label))
        data_ai = dataset[ai_label]
        res_ai = get_score(data_ai)
        ai_scores = [((d["ll"] - d["perturbed_ll"]) / d["perturbed_ll_std"]) if d["perturbed_ll_std"] != 0 else d["ll"] - d["perturbed_ll"] for d in res_ai]
        with open(f"{args.dataset_dir}/{ai_label}_scores.pkl", "wb") as f:
            pickle.dump(ai_scores, f)
    
    # load if f"{human_label}_scores.pkl" is present
    if os.path.exists(f"{args.dataset_dir}/{human_label}_scores.pkl"):
        print(f"Loading {human_label} from pickle")
        with open(f"{args.dataset_dir}/{human_label}_scores.pkl", "rb") as f:
            human_scores = pickle.load(f)
    else:
        print(f"Calculating {human_label} since it is not saved from before")
        dataset = dataset.map(lambda example: strip_whitespace(example, ai_label, human_label))
        data_human = dataset[human_label]
        res_human = get_score(data_human)
        human_scores = [((d["ll"] - d["perturbed_ll"]) / d["perturbed_ll_std"]) if d["perturbed_ll_std"] != 0 else d["ll"] - d["perturbed_ll"] for d in res_human]
        with open(f"{args.dataset_dir}/{human_label}_scores.pkl", "wb") as f:
            pickle.dump(human_scores, f)

    # plot_overlaying_histograms(human_scores, ai_scores, method, human_label, ai_label)

    fpr, tpr, roc_auc = get_roc_metrics(human_scores, ai_scores)
    print("ROC AUC:", roc_auc)
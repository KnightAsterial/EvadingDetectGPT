from customDetectGPT import get_score 
from datasets import load_from_disk, load_dataset
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt


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
    
    # CONFIG
    method = "paraphrased"
    ai_label = "ai_sample"
    human_label = "rephrased_sample"
    dataset = load_from_disk(f"test_out_wiki")
    
    dataset = dataset.map(lambda example: strip_whitespace(example, ai_label, human_label))
    data_human = dataset[human_label]
    data_ai = dataset[ai_label]
    res_human = get_score(data_human)
    # res_human = get_score(dataset["document"][:10])

    res_ai = get_score(data_ai)
    human_scores = [((d["ll"] - d["perturbed_ll"]) / d["perturbed_ll_std"]) if d["perturbed_ll_std"] != 0 else d["ll"] - d["perturbed_ll"] for d in res_human]
    ai_scores = [((d["ll"] - d["perturbed_ll"]) / d["perturbed_ll_std"]) if d["perturbed_ll_std"] != 0 else d["ll"] - d["perturbed_ll"] for d in res_ai]

    plot_overlaying_histograms(human_scores, ai_scores, method, human_label, ai_label)
    fpr, tpr, roc_auc = get_roc_metrics(np.array(human_scores), np.array(ai_scores))
    print("ROC AUC:", roc_auc)
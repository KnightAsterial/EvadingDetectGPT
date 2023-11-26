from customDetectGPT import get_score 
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
sent1 = "Hello typing something to test out the code"
sent2 = "Hi tying to test out now the code"



sent1b = "AsdKj hi, test"
sent2b = 'asdkj hi test'

dataset = load_from_disk("/jagupard25/scr0/wychow/EvadingDetectGPT/data")

sample = "Sexhow railway station was a railway station built to serve the hamlet of Sexhow in North Yorkshire, England. The station was on the North Yorkshire and Cleveland's railway line between Essex and London, which opened in 1857."

res_human = get_score(dataset["summary"][:200])

res_ai = get_score(dataset["generated"][:200])


human_scores = [((d["ll"] - d["perturbed_ll"]) / d["perturbed_ll_std"]) if d["perturbed_ll_std"] != 0 else d["ll"] - d["perturbed_ll"] for d in res_human]
ai_scores = [((d["ll"] - d["perturbed_ll"]) / d["perturbed_ll_std"]) if d["perturbed_ll_std"] != 0 else d["ll"] - d["perturbed_ll"] for d in res_ai]


print(human_scores)
print(ai_scores)
diff = np.array(human_scores) - np.array(ai_scores)
print(diff) # want this to be negative

import matplotlib.pyplot as plt
# plot histogram of diff and save fig
plt.hist(diff, bins=50)
plt.savefig("histogram.png")

# plot histogram of human scores and ai_scores in same plot
plt.clf()
plt.hist(human_scores, bins=50, alpha=0.5, label="human")
plt.hist(ai_scores, bins=50, alpha=0.5, label="ai")
plt.legend(loc='upper right')
plt.savefig("histogram2.png")


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

fpr, tpr, roc_auc = get_roc_metrics(np.array(human_scores), np.array(ai_scores))
print("ROC AUC:", roc_auc)
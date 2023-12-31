# -*- coding: utf-8 -*-
"""CS330Final_EditCountStatistics

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vL_58AQ6rU2n9gDtEOXmh8kA19BihOFu
"""

# ! pip install torch numpy transformers datasets matplotlib tqdm scikit-learn sentencepiece tensorboard sentencepiece nltk

from datasets import load_from_disk
import numpy as np
import matplotlib.pyplot as plt
import nltk
from customDetectGPT import get_score 
# nltk.download('punkt')

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

# def detectgpt_score(example):
#     rephrased_scores = get_score(example["rephrased_sample"])
#     ai_scores = get_score(example["ai_sample"])
#     return {"rephrased_scores": rephrased_scores, "ai_scores": ai_scores}

def split_and_count_edits(example):
    ai_sents = nltk.sent_tokenize(example["ai_sample"])
    rephrased_sents = nltk.sent_tokenize(example["rephrased_sample"])
    result = []
    for (ai_sent, rephrased_sent) in zip(ai_sents, rephrased_sents):
        result.append(edit_distance(ai_sent, rephrased_sent))
    return {'edit_distances': result, "mean_edit_distances": np.mean(result)}


# dataset = load_from_disk('./modeltest_cp6k')
dataset = load_from_disk('./modeltest_cp0_noinnerloop')
res_ai = get_score(dataset["ai_sample"])
res_rephrased = get_score(dataset["rephrased_sample"])

rephrased_scores = [((d["ll"] - d["perturbed_ll"]) / d["perturbed_ll_std"]) if d["perturbed_ll_std"] != 0 else d["ll"] - d["perturbed_ll"] for d in res_rephrased]
ai_scores = [((d["ll"] - d["perturbed_ll"]) / d["perturbed_ll_std"]) if d["perturbed_ll_std"] != 0 else d["ll"] - d["perturbed_ll"] for d in res_ai]

dataset = dataset.add_column("ai_scores", ai_scores)
dataset = dataset.add_column("rephrased_scores", rephrased_scores)

dataset = dataset.map(split_and_count_edits)
print(dataset[:2])

# plt.scatter(dataset["num_edits"], dataset["mean_edit_distances"])
# plt.xlabel("num_edits picked for task")
# plt.ylabel("mean_edit_distance")
# plt.title("Checkpoint 60k")
# plt.show()

# dataset_cp0_wis = load_from_disk('./modeltest_cp0_withinnerstep') # wis = with_inner_step
# dataset_cp0_wis = dataset_cp0_wis.map(split_and_count_edits)


# plt.scatter(dataset_cp0_wis["num_edits"], dataset_cp0_wis["mean_edit_distances"])
# plt.xlabel("num_edits picked for task")
# plt.ylabel("mean_edit_distance")
# plt.title("Checkpoint 0")
# plt.show()

# # plot the difference in scores against the number of edits picked
# plt.clf()
# plt.scatter(dataset["mean_edit_distances"], dataset["rephrased_scores"] - np.array(dataset["ai_scores"]))
# plt.xlabel("mean_edit_distances")
# plt.ylabel("rephrased scores - ai scores")
# plt.title("Checkpoint 60k")
# # save
# plt.savefig("score_vs_edits.png")

# plot the difference in scores against the number of edits picked
plt.clf()
plt.scatter(dataset["mean_edit_distances"], dataset["rephrased_scores"] - np.array(dataset["ai_scores"]))
plt.xlabel("mean_edit_distances")
plt.ylabel("rephrased scores - ai scores")
plt.title("Baseline")
# save
plt.savefig("score_vs_edits_0k.png")


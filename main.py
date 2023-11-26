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

data_human = ["Hamas fighters in black balaclavas and green bandanas are present at all the handovers, along with Red Cross officials wearing white jackets and bibs clearly marked with the Red Cross logo. In several of the clips, the gunmen wave goodbye to the hostages, who appear to have little choice but to wave and smile in return — a response under duress that likely reflects their relief at going home after seven weeks in captivity. Another clip shows the four Thai nationals also being escorted to a Red Cross vehicle and climbing in the back. They, too, appear to feel coerced into waving and giving the ‘thumbs up’ sign to their captors.",
              "Police arrived at the scene on the 1600 block of 6th Avenue East at around 10:30 p.m. There was an unresponsive victim inside the home who died, despite life-saving measures. Officials say the death is suspicious, and are treating it as a homicide, though they do not say there is a risk to the public. The victim will be identified after the family is notified.",
              "The program is called Friday Plans, and they offer everyone a free prescription when one registers under this program. Because of the free prescription you don’t need health insurance or a previous prescription for Generic Viagra to sign up for this program. You don’t even need to talk to a doctor in most states.",
              "In a defiant speech Friday sprinkled with taunts and obscenities aimed at his congressional colleagues, Santos insisted he was “not going anywhere”. But he acknowledged that his time as a member of Congress may soon be coming to an end.",
              "The Red Cross transferred the hostages to Egypt, and they later underwent an initial medical assessment inside Israeli territory and were brought to hospitals in Israel to reunite with their families, according to the IDF."]

# data_ai = ["At every handover, individuals affiliated with Hamas don black balaclavas and green bandanas, standing alongside Red Cross officials clad in white jackets and bibs prominently displaying the Red Cross logo. In various footage, the armed individuals bid farewell to the hostages. The captives, seemingly compelled by the circumstances, reciprocate with waves and smiles—a likely expression of their relief as they head home after enduring seven weeks of captivity. In one specific clip, the four Thai nationals are seen being led to a Red Cross vehicle, where they reluctantly wave and give a 'thumbs up' gesture to their captors.",
#            "At approximately 10:30 p.m., law enforcement responded to the incident on the 1600 block of 6th Avenue East. Within the residence, an unresponsive individual was discovered, and despite efforts to save their life, the victim succumbed. Authorities are deeming the death suspicious and treating it as a homicide; however, they have not indicated any perceived threat to public safety. The victim's identity will be disclosed following notification of the family.",
#            "Named Friday Plans, the initiative provides a complimentary prescription to all participants upon registration. Enrollment in this program does not necessitate health insurance or a prior prescription for Generic Viagra. In fact, in most states, speaking to a doctor is not even a requirement to join.",
#            "In a bold address on Friday, laced with jibes and explicit language directed at his fellow lawmakers, Santos asserted his resilience, stating firmly that he was 'here to stay.' However, he did concede that his tenure as a congressional member might be drawing to a close.",
#            "The hostages were handed over to Egypt by the Red Cross. Subsequently, they underwent an initial medical evaluation within Israeli territory before being transported to hospitals in Israel to be reunited with their families, as reported by the IDF."]

data_ai = ["In a landmark decision, the government announces a nationwide initiative to enhance digital literacy. The program aims to provide accessible online education to bridge the digital divide. With a focus on remote areas, this initiative aligns with broader efforts to ensure equal opportunities and connectivity for all citizens in the digital age.",
           "NASA's latest rover, Perseverance, uncovers compelling evidence of ancient microbial life on Mars. Analyzing rock samples, scientists identified organic molecules and patterns consistent with microbial fossils. The findings mark a significant step in understanding Mars' past and the potential for extraterrestrial life, sparking renewed enthusiasm for future exploration.",
           "In a diplomatic breakthrough, neighboring countries announce a historic peace agreement, ending years of conflict. Leaders express optimism for a new era of cooperation and mutual prosperity. The accord is celebrated globally as a testament to diplomatic efforts and a step towards fostering stability in the region.",
           "Renowned tech company releases an innovative AI-powered personal assistant, set to revolutionize daily tasks. The advanced system adapts to user preferences, streamlining schedules, and providing real-time information. Early reviews praise its intuitive interface and efficiency, marking a significant advancement in artificial intelligence applications for personal use.",
           "Amid rising concerns about cybersecurity, a major tech conglomerate unveils a state-of-the-art encryption system. The new technology promises enhanced protection for user data, making it significantly more challenging for unauthorized access. Industry experts applaud the development, anticipating it will set a new standard for digital security in an increasingly interconnected world."]

from datasets import load_dataset
dataset = load_dataset("aadityaubhat/GPT-wiki-intro", split="train[:100]")
data_human = dataset["wiki_intro"]
data_ai = dataset["generated_intro"]
res_human = get_score(data_human)
# res_human = get_score(dataset["document"][:10])

res_ai = get_score(data_ai)
# res_ai = get_score(dataset["generated"][:10])

print(res_human)
print(res_ai)
human_scores = [((d["ll"] - d["perturbed_ll"]) / d["perturbed_ll_std"]) if d["perturbed_ll_std"] != 0 else d["ll"] - d["perturbed_ll"] for d in res_human]
ai_scores = [((d["ll"] - d["perturbed_ll"]) / d["perturbed_ll_std"]) if d["perturbed_ll_std"] != 0 else d["ll"] - d["perturbed_ll"] for d in res_ai]

print("HUMAN", human_scores)
print("AI", ai_scores)
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


# def get_roc_metrics(real_preds, sample_preds):
#     fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
#     roc_auc = auc(fpr, tpr)
#     return fpr.tolist(), tpr.tolist(), float(roc_auc)


# def get_precision_recall_metrics(real_preds, sample_preds):
#     precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
#     pr_auc = auc(recall, precision)
#     return precision.tolist(), recall.tolist(), float(pr_auc)

# fpr, tpr, roc_auc = get_roc_metrics(np.array(human_scores), np.array(ai_scores))
# print("ROC AUC:", roc_auc)
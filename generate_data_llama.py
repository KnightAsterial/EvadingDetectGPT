from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM
# import nltk
# nltk.download('punkt')

# dataset = load_dataset("EdinburghNLP/xsum", split="train[:300]")
dataset = load_dataset("aadityaubhat/GPT-wiki-intro", split="train[:10]")
batch_size = 32

# load in model for paraphrasing
# model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws", torch_dtype=torch.bfloat16).to("cuda")
# tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")

# map summary of dataset to add prefix and suffix
# def format_dataset(example):
#     example["formatted"] = "paraphrase: " + example["document"] + " </s>"
#     # example["formatted"] = "paraphrase: " + example["summary"] + " </s>"
#     return example

def format_dataset(example):
    example["formatted"] = "Paraphrase: " + example["wiki_intro"]
    return example

def tokenize(example):
    return tokenizer(example["formatted"], return_tensors="pt", padding=True)

dataset = dataset.filter(lambda example: len(tokenizer(example["wiki_intro"])["input_ids"]) < 254)
dataset = dataset.map(format_dataset)

#---------
# def process(example):
#     lines = example['document'].split('\n')
#     new_lines = []
#     for line in lines:
#         line = line.strip()
#         if len(line) == 0:
#             new_lines.append(line)
#         else:
#             sents = nltk.sent_tokenize(line)

#             formatted_sents = ["paraphrase: " + sent + " </s>" for sent in sents]
#             token_dict = tokenizer(formatted_sents, padding=True, return_tensors="pt")
#             output = model.generate(
#                 token_dict["input_ids"].to("cuda"),
#                 attention_mask=token_dict["attention_mask"].to("cuda"),
#                 max_length=256,
#                 do_sample=True,
#                 top_k=200,
#                 top_p=0.95,
#                 num_return_sequences=1
#             )
#             paraphrased_sents = tokenizer.batch_decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
#             # paraphrased_sents = [paraphrased_sent.split() for paraphrased_sent in paraphrased_sents]
#             # new_lines.append(' '.join(paraphrased_sents))
#             new_lines.append(paraphrased_sents[0])
#     return {"generated": '\n'.join(new_lines)}

# dataset = dataset.map(process, remove_columns=['summary'])
#--------

dataset = dataset.with_format("torch")
dataset = dataset.map(tokenize, batched=True, batch_size=batch_size)


def paraphrase(example):
    # print("---\n", example["input_ids"])
    output = model.generate(
        example["input_ids"].to("cuda"),
        attention_mask=example["attention_mask"].to("cuda"),
        max_new_tokens=512,
        do_sample=True,
        top_k=200,
        top_p=0.95,
        num_return_sequences=1
    )
    return {"generated": tokenizer.batch_decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)}

# dataset = dataset.map(paraphrase, remove_columns=["input_ids", "attention_mask", "formatted", "id"], batched=True, batch_size=batch_size)
dataset = dataset.map(paraphrase, batched=True, batch_size=batch_size)
cols_to_remove = dataset.column_names
cols_to_remove.remove("wiki_intro")
cols_to_remove.remove("generated")
dataset = dataset.remove_columns(cols_to_remove)
print(dataset[:2])
dataset.save_to_disk("/jagupard26/scr0/wychow/EvadingDetectGPT/data")

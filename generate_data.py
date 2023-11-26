from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

dataset = load_dataset("EdinburghNLP/xsum", split="train[:100]")
batch_size = 32

# load in model for paraphrasing
model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws", torch_dtype=torch.bfloat16).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")

# map summary of dataset to add prefix and suffix
def format_dataset(example):
    example["formatted"] = "paraphrase: " + example["document"] + " </s>"
    # example["formatted"] = "paraphrase: " + example["summary"] + " </s>"
    return example

def tokenize(example):
    return tokenizer(example["formatted"], return_tensors="pt", padding=True)

dataset = dataset.filter(lambda example: len(tokenizer(example["document"])["input_ids"]) < 254)
dataset = dataset.map(format_dataset, remove_columns=['summary'])
# dataset = dataset.map(format_dataset, remove_columns=['document'])
dataset = dataset.with_format("torch")
dataset = dataset.map(tokenize, batched=True, batch_size=batch_size)


def paraphrase(example):
    # print("---\n", example["input_ids"])
    output = model.generate(
        example["input_ids"].to("cuda"),
        attention_mask=example["attention_mask"].to("cuda"),
        max_length=256,
        do_sample=True,
        top_k=200,
        top_p=0.95,
        num_return_sequences=1
    )
    return {"generated": tokenizer.batch_decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)}

dataset = dataset.map(paraphrase, remove_columns=["input_ids", "attention_mask", "formatted", "id"], batched=True, batch_size=batch_size)
print(dataset[:2])
dataset.save_to_disk("/jagupard25/scr0/wychow/EvadingDetectGPT/data")

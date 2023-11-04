import numpy as np
import re
import torch
import tqdm
import functools
from multiprocessing.pool import ThreadPool
import time
import globals

def load_base_model():
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    try:
        globals.MASK_MODEL.cpu()
    except NameError:
        pass
    globals.BASE_MODEL.to(globals.device)
    print(f'DONE ({time.time() - start:.2f}s)')

def load_mask_model():
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    globals.BASE_MODEL.cpu()
    globals.MASK_MODEL.to(globals.device)
    print(f'DONE ({time.time() - start:.2f}s)')

# functions to support get_perturbation_results
def perturb_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = globals.chunk_size

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    print("masked")
    raw_fills = replace_masks(masked_texts)
    print("replaced")
    extracted_fills = extract_fills(raw_fills)
    print("extracted")
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
    print("perturbed")

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

    return perturbed_texts

def tokenize_and_mask(text, span_length, pct, ceil_pct=False, buffer_size=1):
    print("starting tokenize_and_mask")
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text

def replace_masks(texts):
    n_expected = count_masks(texts)
    print('entered replace masks')
    stop_id = globals.MASK_TOKENIZER.encode(f"<extra_id_{max(n_expected)}>")[0]
    print('stopid done')
    tokens = globals.MASK_TOKENIZER(texts, return_tensors="pt", padding=True).to(globals.device)
    print('tokens done)')
    outputs = globals.MASK_MODEL.generate(**tokens, max_length=150, do_sample=True, top_p=globals.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    print('outputs done')
    return globals.MASK_TOKENIZER.batch_decode(outputs, skip_special_tokens=False)

def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    pattern = re.compile(r"<extra_id_\d+>")
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts

def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]

# Get the log likelihood of each text under the globals.BASE_MODEL
def get_ll(text):
    with torch.no_grad():
        tokenized = globals.BASE_TOKENIZER(text, return_tensors="pt").to(globals.device)
        labels = tokenized.input_ids
        return -globals.BASE_MODEL(**tokenized, labels=labels).loss.item()


def get_lls(texts):
    # use GPT2_TOKENIZER to get total number of tokens
    total_tokens = sum(len(globals.GPT2_TOKENIZER.encode(text)) for text in texts)
    globals.API_TOKEN_COUNTER += total_tokens * 2  # multiply by two because OpenAI double-counts echo_prompt tokens

    pool = ThreadPool(globals.batch_size)
    return pool.map(get_ll, texts)


def get_perturbation_results(text, span_length=10, n_perturbations=1, n_perturbation_rounds=1, pct_words_masked=0.15):
    load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    # TODO
    # text = data["original"]

    perturb_fn = functools.partial(perturb_texts, span_length=span_length, pct=pct_words_masked)

    p_text = perturb_fn([x for x in text for _ in range(n_perturbations)])
    for _ in range(n_perturbation_rounds - 1):
        try:
            p_text = perturb_fn(p_text)
        except AssertionError:
            break

    assert len(p_text) == len(text) * n_perturbations, f"Expected {len(text) * n_perturbations} perturbed samples, got {len(p_text)}"

    for idx in range(len(text)):
        results.append({
            "text": text[idx],
            "perturbed": p_text[idx * n_perturbations: (idx + 1) * n_perturbations]
        })

    load_base_model()

    for res in tqdm.tqdm(results, desc="Computing log likelihoods"):
        p_ll = get_lls(res["perturbed"])
        res["ll"] = get_ll(res["text"])
        res["all_perturbed_ll"] = p_ll
        res["perturbed_ll"] = np.mean(p_ll)
        res["perturbed_ll_std"] = np.std(p_ll) if len(p_ll) > 1 else 1

    return results

# The Function
def get_score(texts):
    # run perturbation experiments
    n_perturbations = globals.n_perturbations # detectGPT uses 100 perturbations for evaluation
    span_length = 2
    perturbation_results = get_perturbation_results(texts, span_length, n_perturbations)
    return perturbation_results
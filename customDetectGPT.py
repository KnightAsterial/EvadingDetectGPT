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
    raw_fills = replace_masks(masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

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
    print("COUNT MASKS", n_expected)
    stop_id = globals.MASK_TOKENIZER.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = globals.MASK_TOKENIZER(texts, return_tensors="pt", padding=True).to(globals.device)
    outputs = globals.MASK_MODEL.generate(**tokens, max_length=150, do_sample=True, top_p=globals.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
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


def get_perturbation_results(text, span_length=2, n_perturbations=100, n_perturbation_rounds=1, pct_words_masked=0.30):
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


# Five off-licences in Belfast\'s Holyland area are to close "voluntarily" for a number of hours on St Patrick\'s Day.', 'perturbed': [
# Five off-licences in Belfast\'s Holyland area are to close "voluntarily" for a number of hours on St Patrick\'s Day.', 'Five off-licences in Belfast\'s Holyland area are to close "
# voluntarily" for a number of hours on St Patrick\'s Day.', 
# 'Five off-licences in Belfast\'s Holyland area are to close "voluntarily" for a number of hour
# s on St Patrick\'s Day.', 'Five off-licences in Belfast\'s Holyland area are to close "voluntarily" for a number of hours on St Patrick\'s Day.', 'Five o
# ff-licences in Belfast\'s Holyland area are to close "voluntarily" for a number of hours on St Patrick\'s Day.', 'Five off-licences in Belfast\'s Holylan

[-1.87830793e-01, -7.95079282e-02, 6.95317335e-01, 2.68923465e-01, 7.63272412e-01, -1.63719705e+00, 5.66994123e-01, -1.38681472e+00, -6.39864170e-01, -3.33702589e-02, -1.48738759e+00, 8.28727122e-01, -1.10179410e+00, 7.14470436e-02, 1.05039812e+00, -2.26278116e+00, -4.87680276e-01, 1.21698119e-01, -1.06460906e+00, -3.94768913e-01, -4.68086808e-02, 5.03873056e-01, -7.28837171e-02, 5.04078960e-02, -8.57051033e-01, 9.41557367e-01, -6.23218411e-02, -2.30720146e+00, 1.07910124e-03, -7.19540940e-01, 5.44596323e-01, 7.25898589e-02, 3.23446115e-01, -1.24109685e-02, -1.52350255e+00, 1.08032323e+00, 3.25601016e-03, 4.95832515e-02, -5.43142766e-02, 3.43989545e-01, -8.66545531e-01, 2.76496207e-01, -7.05446370e-01, 7.80738393e-01, -2.76893791e-01, 6.85578475e-01, -1.82800860e+00, -6.43354586e-01, -4.52457962e-01, 3.69611655e-01, 1.11981183e-01, 1.31634310e+00, -1.24683267e-01, 9.83487443e-01, -2.04513469e+00, 5.34459788e-01, -2.31090142e-03, -1.53202470e-01, -1.33221532e+00, -1.17617528e+00, -2.57698109e+00, 4.00712253e-01, 1.19721214e-01, 4.36524947e-01, 1.78496147e+00, 3.69966103e-01, 1.31674972e+00, 1.18823852e-01, -1.19501549e+00, -8.06919197e-01, 1.05437699e+00, -5.02849364e-01, -5.25834850e-01, -5.11988984e-01, 4.35268611e-01, -1.33778994e-01, -2.91044188e-01, 9.40357611e-02, 1.41458062e+00, -8.55126876e-02, -7.03421593e-01, 3.40066094e-02, -1.50336076e+00, 1.50404768e-01, -3.69491325e-01, -1.07276529e+00, 4.93842666e-01, 7.46017551e-01, 3.54760797e-01, 9.92580415e-01, 2.56196942e-01, 3.15334049e-01, 5.07244759e-01, 1.97487116e+00, -5.30793288e-01]
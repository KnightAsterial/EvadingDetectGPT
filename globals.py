import transformers
import torch

## CONFIGURATION
mask_filling_model_name = "t5-large"
base_model_name = "EleutherAI/gpt-neo-125m"
device = "cpu"
# cache_dir = "~/.cache"
int8 = False
bf16 = True

chunk_size = 20
batch_size = 50
mask_top_p = 1.0

n_perturbations = 1#00 

#####################################################################



global GPT2_TOKENIZER 
GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2')

global MASK_MODEL
int8_kwargs = {}
half_kwargs = {}
if int8:
    int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
elif bf16:
    bf16_kwargs = dict(torch_dtype=torch.bfloat16)
MASK_MODEL = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, **int8_kwargs, **bf16_kwargs)
try:
    n_positions = MASK_MODEL.config.n_positions
except AttributeError:
    n_positions = 512

MASK_TOKENIZER = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions)


base_model_kwargs = {"torch_dtype": torch.float16}
BASE_MODEL = transformers.AutoModelForCausalLM.from_pretrained(base_model_name, **base_model_kwargs)

BASE_TOKENIZER = transformers.AutoTokenizer.from_pretrained(base_model_name)
BASE_TOKENIZER.pad_token_id = BASE_TOKENIZER.eos_token_id






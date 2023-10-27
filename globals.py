import transformers


global cache_dir
cache_dir = "~/.cache"

global GPT2_TOKENIZER 
GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
from datasets import load_dataset
from datasets import Features, Sequence, Value
import llm_utils
import numpy as np

torch.cuda.empty_cache()

login(token="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")
cache_dir='/home/chawak/models/huggingface'
n=3
split='train'

#initializing the model and tokenizer
tokenizer, model = llm_utils.get_model_tokenizer()

#load the dataset
dataset_path=f'../data/{n}_blocks/SFT_full_{split}_{n}_blocks'
data=load_dataset("csv",data_files={split:dataset_path})
data=data.remove_columns(['Unnamed: 0'])
print(f'Loaded HuggingFace dataset : {data}')

#helper function for tokenizing
def tokenize_and_mask_function(examples):
    #concatinate prompt and gold plan within our template
    merged_inputs=[f"{p[:-2]}{g[6:]}" for p,g in zip(examples['prompt'],examples['gold_plan'])]

    #tokenize concatinated input
    tokenized=tokenizer(
        merged_inputs,
        truncation=True,
        padding='max_length',
        padding_side='right',
        max_length=1200
    )

    #tokenize ONLY the prompts and get their lengths
    tokenized_prompts=tokenizer(
        examples['prompt'],
        truncation=False
    )

    #print(f'Tokenized prompts: {tokenized_prompts}')
    prompt_lens = [len(ptoken) for ptoken in tokenized_prompts['input_ids']]
    #print("Lengths of tokenized prompts", prompt_lens)
    
    #estimating input token sequence lengths
    merged_input_lens=[len(merged) for merged in tokenized['input_ids']]
    sorted_lens=sorted(merged_input_lens, reverse=True)
    #print('Highest lengths of merged-input token sequence:',sorted_lens[:5])
    #print("Number of prompts tokenized",len(prompt_lens))
    
    #masking prompt tokens for the labels 
    labels=[]
    for input_ids, prompt_length in zip(tokenized['input_ids'],prompt_lens):
        label=input_ids.copy()
        #mask prompt tokens as -100 & adjustment for prompt template
        label[:prompt_length-4]=[-100]*prompt_length
        label=label[:-4]
        labels.append(label)    
    tokenized['labels']=labels

    tokenized['input_ids'] = [np.array(ids, dtype=np.int64) for ids in tokenized['input_ids']]
    tokenized['labels']=[np.array(labels, dtype=np.int64) for labels in tokenized['labels']]
    return tokenized

#map tokenizer function to our dataset
tokenized_data=data.map(tokenize_and_mask_function,batched=True)
new_features = Features({
    "prompt": Value("string"),
    "gold_plan": Value("string"),
    "input_ids": Sequence(Value("int64")),
    "attention_mask": Sequence(Value("int64")),
    "labels": Sequence(Value("int64")),
})

tokenized_data = tokenized_data.cast(new_features)

#sanity checks on tokenized data
print(f'Data dictonary after tokenization: {tokenized_data}')
#decoding for merged-text
encoded_text=tokenized_data[split][0]['input_ids']
print(f'Length of encoded merged-text : {len(encoded_text)}')
decoded_text=tokenizer.decode(encoded_text, skip_special_tokens=False)
print(f'Length of decoded merged-text : {len(encoded_text)}')
print(f'Decoded merged-text: {decoded_text}')
#deocding for lables
encoded_text=tokenized_data[split][0]['labels']
print(f'Length of encoded labels : {len(encoded_text)}')
decoded_text=tokenizer.decode(
    [token_id for token_id in encoded_text if token_id!=-100],
    skip_special_tokens=True)
print(f'Decoded labels: {decoded_text}')

#save to file
#tokenized_data.save_to_disk(f"/srv/chawak/planning-with-llms/data/{n}_blocks/tokenized_dataset")
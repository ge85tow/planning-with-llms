import sys
sys.path.append("/srv/chawak/planning-with-llms/src/")
from shared import llm_utils


import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
#multi-GPU set-up
# os.environ['TORCH_DISTRIBUTED_USE_DTENSOR'] = '0'


# from accelerate import Accelerator #multi-GPU set-up
import torch
# torch._dynamo.disable()  #multi-GPU set-up

from huggingface_hub import login
import sys
import yaml
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model,PeftModel#,AdaLoraModel, AdaLoraConfig
from transformers import DataCollatorWithPadding
from datasets import concatenate_datasets
import pandas as pd
from datasets import load_from_disk

#multi-gpu set-up
# accelerator = Accelerator()

login(token="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")

#get config file
with open("/srv/chawak/planning-with-llms/src/sft/config.yaml", "r") as f:
    cfg=yaml.safe_load(f)

# output_dir=cfg['training']['output_dir']+

#set environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

#get model
base_model,tokenizer=llm_utils.get_model_tokenizer()

#LORA config 
peft_args=LoraConfig(
    r=int(cfg['peft']['r']),
    lora_alpha=int(cfg['peft']['lora_alpha']),
    lora_dropout=float(cfg['peft']['lora_dropout']),
    task_type=cfg['peft']['task_type'],
    target_modules=list(cfg['peft']['target_modules'])
)

#ADALORA Config
# peft_args=AdaLoraConfig(
#     peft_type=cfg['adalora']['peft_type'],
#     init_r=int(cfg['peft']['r']),
#     target_r=int(cfg['peft']['target_r'])
#     lora_alpha=int(cfg['peft']['lora_alpha']),
#     lora_dropout=float(cfg['peft']['lora_dropout']),
#     task_type=cfg['peft']['task_type'],
#     target_modules=list(cfg['peft']['target_modules'])
# )

#wrapper for gemma-3
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    max_length=1200,
    padding='max_length',
)

#change for pad token id
def wrapped_collator(features):
    batch = data_collator(features)
    # Replace pad token ids in labels with -100
    if "labels" in batch:
        labels = batch["labels"]
        # Replace pad token ids in labels with -100
        labels = torch.where(labels == tokenizer.pad_token_id, torch.full_like(labels, -100), labels)
        batch["labels"] = labels
    return batch

#print(f'Tokenizer max length: {tokenizer.model_max_length}')
#print(f'Default data collator max length : {data_collator.max_length}')

cpt=36320

#resume from a checkpoint
checkpoint_path=f"/srv/chawak/planning-with-llms/results/SFT/training/training_05-06/checkpoint-{cpt}"

def get_train_args():
    training_args=SFTConfig(
        output_dir=cfg['training']['output_dir'],
        overwrite_output_dir=False, #resume from checkpoint
        resume_from_checkpoint=checkpoint_path, #resume from checkpoint
        num_train_epochs=int(cfg['training']['num_train_epochs']),
        per_device_train_batch_size= int(cfg['training']['per_device_train_batch_size']),
        per_device_eval_batch_size=int(cfg['training']['per_device_eval_batch_size']),
        gradient_accumulation_steps=int(cfg['training']['gradient_accumulation_steps']),
        learning_rate=float(cfg['training']['learning_rate']),
        weight_decay=float(cfg['training']['weight_decay']),
        warmup_ratio=float(cfg['training']['warmup_ratio']),
        eval_strategy=cfg["training"]["evaluation_strategy"],
        logging_strategy=cfg['training']['logging_strategy'],
        save_strategy=cfg["training"]["save_strategy"],
        fp16 = bool(cfg['training']['fp16']),
        bf16 = bool(cfg['training']['bf16']),
        report_to=cfg["training"]["report_to"],
        label_names=['labels']
    )
    return training_args

def train(train_data,eval_data,model):

    training_args=get_train_args()
    trainer=SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=wrapped_collator, #change for pad token id
    eval_dataset=eval_data,
    )

    #trainer_stats=trainer.train()

    #resume from checkpoint
    print(f'Resuming training from checkpoint {training_args.resume_from_checkpoint}')
    trainer_stats=trainer.train(resume_from_checkpoint=checkpoint_path)

    #debug
    #inspect_lora_weights(model)

    #path
    path=trainer.args.output_dir
    #saving only adapters    
    trainer.model.save_pretrained(path)
    #saving trainer state
    trainer.save_state()

    print(f'Saving trained adapter & trainer state to{path}')
    return trainer_stats


def load_tokenized_data(n):

    data_dir=f"/srv/chawak/planning-with-llms/data/{n}_blocks"
    #load train dataset
    split='train'
    data_path=f'{data_dir}/tokenized_dataset/{split}'

    train_data=load_from_disk(data_path)

    #load evaluation dataset
    split='val'
    data_path=f'{data_dir}/tokenized_dataset/{split}'

    eval_data=load_from_disk(data_path)
    #eval_data=eval_data.select(range(1))
    #print(f'The eval data is:{eval_data}')
    return train_data, eval_data


def main():
    
    #fetch train and eval data
    three_train, three_eval = llm_utils.load_tokenized_data(3)
    four_train, four_eval = llm_utils.load_tokenized_data(4)

    #combine three and four block dataset to one
    train_data=concatenate_datasets([three_train,four_train])
    eval_data= concatenate_datasets([three_eval,four_eval])
    # #sanity-check
    # print(f'length of train data : {len(train_data)}')
    # print(f'length of eval data: {len(eval_data)}')
 
    #fetch LORA model
    base_model.gradient_checkpointing_enable()
    model=get_peft_model(base_model,peft_args)
    

    #multi-GPU set-up
    # print(f"Using device: {accelerator.device}")
    # print(f"Number of processes: {accelerator.num_processes}")
    # for i in range(torch.cuda.device_count()):
    #     print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    #     print(f"Supports bf16: {torch.cuda.is_bf16_supported()}")
    # model, train_data, eval_data = accelerator.prepare(model, train_data, eval_data)

    #start training and get stats
    stats = train(train_data=train_data, eval_data=eval_data, model=model)

    #logging epoch-wise stats
    print(f'Train epoch-wise stats: {stats}')
    

main()
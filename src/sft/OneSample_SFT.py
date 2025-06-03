import torch
from huggingface_hub import login
import llm_utils
import yaml
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model,PeftModel#, merge_adapter
from transformers import DataCollatorWithPadding #default_data_collator

login(token="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")
n=3
#get config file
with open("/srv/chawak/planning-with-llms/src/config.yaml", "r") as f:
    cfg=yaml.safe_load(f)

output_dir=cfg['training']['output_dir']+'one_sample'

#set environment variables
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

#get model
base_model,tokenizer=llm_utils.get_model_tokenizer()
base_model.gradient_checkpointing_enable()

#LORA config 
peft_args=LoraConfig(
    r=int(cfg['peft']['r']),
    lora_alpha=int(cfg['peft']['lora_alpha']),
    lora_dropout=float(cfg['peft']['lora_dropout']),
    task_type=cfg['peft']['task_type'],
    target_modules=list(cfg['peft']['target_modules'])
)
#wrapper for gemma-3
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    max_length=1200,
    padding='max_length',
)

#print(f'Tokenizer max length: {tokenizer.model_max_length}')
#print(f'Default data collator max length : {data_collator.max_length}')

def get_train_args(n):
    training_args=SFTConfig(
        output_dir=output_dir,
        num_train_epochs=int(cfg['training']['num_train_epochs']),
        #per_device_train_batch_size= int(cfg['training']['per_device_train_batch_size']),
        #per_device_eval_batch_size=int(cfg['training']['per_device_eval_batch_size']),
        #gradient_accumulation_steps=int(cfg['training']['gradient_accumulation_steps']),
        per_device_train_batch_size= 1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
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

'''def inspect_lora_weights(model):
    count = 0
    for name, param in list(model.named_parameters())[:5]:
        if "lora_B" in name:
            count += 1
            mean = param.data.mean().item()
            std = param.data.std().item()
            print(f"{name} → mean: {mean:.6f}, std: {std:.6f}")
    print(f"\n✅ Total lora_B layers found: {count}")
    return'''

def train(n,train_data,eval_data,model):

    training_args=get_train_args(n)
    trainer=SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    data_collator=data_collator,
    eval_dataset=eval_data,
    )

    trainer_stats=trainer.train()

    #debug
    #inspect_lora_weights(model)

    path=f"{output_dir}"    

    #saving only adapters
    trainer.model.save_pretrained(path)

    print(f'Saving trained adapter to{path}')
    return trainer_stats


def main(n):
    
    #fetch train and eval data
    train_data, eval_data = llm_utils.load_tokenized_data(n)
    
    #one sample runs
    train_data=train_data.select([0])
    eval_data=train_data

    print(f'Train data is: {train_data[0]["prompt"],train_data[0]["gold_plan"]}')
    print(f'Eval data is: {eval_data[0]["prompt"],eval_data[0]["gold_plan"]}')

    #fetch LORA model
    model=get_peft_model(base_model,peft_args)
    model.gradient_checkpointing_enable()
    stats = train(n=n, train_data=train_data, eval_data=eval_data, model=model)
    print(f'Train iteration stats: {stats}')

    # for name, param in model.named_parameters():
    #     if "q_proj.weight" in name:
    #         print(f"{name} → mean: {param.data.mean():.6f}, std: {param.data.std():.6f}")

main(n)
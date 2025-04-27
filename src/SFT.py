from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import llm_utils
import yaml
from transformers import TrainingArguments, AdamW
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model

login(token="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")
cache_dir='/home/chawak/models/huggingface'

#get model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

model,tokenizer=llm_utils.get_model_tokenizer()

#load dataset
n=3
split='train'
data_path=f'../data/{n}_blocks/tokenized_dataset/{split}'
train_data=load_from_disk(data_path)
train_data

#load evaluation dataset
n=3
split='val'
data_path=f'../data/{n}_blocks/tokenized_dataset/{split}'
eval_data=load_from_disk(data_path)
eval_data

#get config file
with open("config.yaml", "r") as f:
    cfg=yaml.safe_load(f)


#LORA config 
peft_args=LoraConfig(
    r=cfg['peft']['r'],
    lora_alpha=cfg['peft']['lora_alpha'],
    lora_dropout=cfg['peft']['lora_dropout'],
    task_type=cfg['peft']['task_type']
)

#get LORA model
lora_model=get_peft_model(model,peft_args)
lora_layers=lora_model.parameters()

training_args=SFTConfig(
    output_dir=cfg['training']['output_dir']+f'/{n}_blocks',
    num_train_epochs=int(cfg['training']['num_train_epochs']),
    per_device_train_batch_size= int(cfg['training']['per_device_train_batch_size']),
    gradient_accumulation_steps=int(cfg['training']['gradient_accumulation_steps']),
    learning_rate=float(cfg['training']['learning_rate']),
    weight_decay=float(cfg['training']['weight_decay']),
    warmup_ratio=float(cfg['training']['warmup_ratio']),
    #adam_epsilon=cfg['training']['adam_epsilon'],
    #optim=cfg['training']['optim'],
    logging_steps=int(cfg['training']['logging_steps']),
    save_steps=int(cfg['training']['save_steps']),
    eval_steps=int(cfg['training']['eval_steps']),
    evaluation_strategy=cfg["training"]["evaluation_strategy"],
    save_strategy=cfg["training"]["save_strategy"],
    fp16=bool(cfg["training"]["fp16"]),
    bf16=bool(cfg["training"]["bf16"]),
    report_to=cfg["training"]["report_to"]
)

#define optimizer args dictionary
optimizer = {
    'params': lora_layers,
    'lr': float(cfg['training']['learning_rate']),
}

trainer=SFTTrainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_args,
    #optimizer_cls_and_kwargs=(AdamW,optimizer)
)

trainer.train()
metrics = trainer.evaluate()
print(f'Evaluation metrics after training {metrics}')
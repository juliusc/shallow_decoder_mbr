import time
import torch
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoTokenizer, GenerationConfig,
                         MarianMTModel, Trainer, Seq2SeqTrainer, DataCollatorForSeq2Seq,
                         TrainingArguments, Seq2SeqTrainingArguments)
from datasets import load_dataset, load_from_disk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = AutoConfig.from_pretrained("Helsinki-NLP/opus-mt-cs-en")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-cs-en")
model = MarianMTModel(config).to(device)

train_ds = load_from_disk("dataset/train")
validation_ds = load_from_disk("dataset/validation")

# For actual training
training_args = Seq2SeqTrainingArguments(
    output_dir="models/base_cs-en",
    learning_rate=5e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=100,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # evaluation_strategy="steps",
    # eval_steps=10,
    # max_steps=50,
    # save_strategy="steps",
    # save_steps=100000,
    auto_find_batch_size=True,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    fp16=True,
    group_by_length=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer),
    train_dataset=train_ds,
    eval_dataset=validation_ds,
)
trainer.train()
import pdb; pdb.set_trace()
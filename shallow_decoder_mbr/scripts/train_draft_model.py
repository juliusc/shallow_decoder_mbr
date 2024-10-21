import time
import torch
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoTokenizer, GenerationConfig,
                         MarianMTModel, Trainer, Seq2SeqTrainer, DataCollatorForSeq2Seq,
                         TrainingArguments, Seq2SeqTrainingArguments)
from datasets import load_dataset, load_from_disk

MAX_GENERATION_LENGTH = 256
NUM_SAMPLES = 8
SAMPLE_GENERATION_CONFIG = GenerationConfig(
    max_length=MAX_GENERATION_LENGTH,
    num_return_sequences=NUM_SAMPLES,
    do_sample=True,
    top_k=0,
    epsilon_cutoff=0.02
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-cs-en")
config = AutoConfig.from_pretrained("Helsinki-NLP/opus-mt-cs-en")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-cs-en")

new_model_config_dict = config.to_dict()
new_model_config_dict["decoder_layers"] = 1
new_model_config_dict["share_encoder_decoder_embeddings"] = False
new_model_config_dict["tie_word_embeddings"] = False
new_model_config = type(config).from_dict(new_model_config_dict)

new_model = MarianMTModel(new_model_config).to(device)
new_model.generation_config = model.generation_config
del model

for param in new_model.model.encoder.parameters():
    param.requires_grad = False

train_ds = load_from_disk("dataset/train")
validation_ds = load_from_disk("dataset/validation")

# For actual training
training_args = Seq2SeqTrainingArguments(
    output_dir="models/draft_cs-en",
    learning_rate=5e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=100,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_steps=100000,
    auto_find_batch_size=True,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    fp16=True,
    group_by_length=True,
)

trainer = Seq2SeqTrainer(
    model=new_model,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer),
    train_dataset=train_ds,
    eval_dataset=validation_ds,
)
trainer.train()

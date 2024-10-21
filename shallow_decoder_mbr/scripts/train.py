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
new_model_config = type(config).from_dict(new_model_config_dict)

# new_model = MarianMTModel(new_model_config).to(device)
new_model = MarianMTModel.from_pretrained("model_first_try/draft_cs-en/checkpoint-453125",use_safetensors=True).to(device)
# new_model = model

new_model.model.encoder = model.model.encoder
# new_model.model.decoder.layers[0] = model.model.decoder.layers[0]
new_model.model.decoder.embed_tokens = model.model.decoder.embed_tokens
new_model.generation_config = model.generation_config

for param in new_model.model.encoder.parameters():
    param.requires_grad = False

dataset = load_dataset("Helsinki-NLP/opus-100", "cs-en")
train_ds = dataset["train"]
validation_ds = dataset["validation"]

# def convert_dataset(dataset):
#     def get_length(x):
#         x["length"] = len(x["translation"]["cs"]) + len(x["translation"]["en"])
#         return x

#     def tokenize(x):
#         return tokenizer(text=x["translation"]["cs"], text_target=x["translation"]["en"], truncation=True)

#     return dataset.map(get_length).sort("length").map(tokenize, remove_columns=["length", "translation"])

def reverse_dataset(dataset):
    def get_length(x):
        x["length"] = -(len(x["input_ids"]) + len(x["labels"]))
        return x

    return dataset.map(get_length).sort("length").map(lambda x:x, remove_columns=["length"])

# new_model.to("cpu")

# train_ds = convert_dataset(train_ds)
# validation_ds = convert_dataset(validation_ds)
# train_ds.save_to_disk("dataset/train")
# validation_ds.save_to_disk("dataset/validation")
# import pdb; pdb.set_trace()

train_ds = load_from_disk("dataset/train")
validation_ds = load_from_disk("dataset/validation")
# train_ds = train_ds.select(range(100))
# validation_ds = validation_ds.select(range(10))
# train_ds = reverse_dataset(train_ds)
# validation_ds = reverse_dataset(validation_ds)

def compute_metrics(eval_pred):
    labels = tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=True)
    predictions = tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=True)
    import pdb; pdb.set_trace()

# For actual training
training_args = Seq2SeqTrainingArguments(
    # output_dir="model_try_again",
    output_dir="model_first_try/draft_cs-en",
    learning_rate=5e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=100,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # evaluation_strategy="epoch",
    # eval_steps=1,
    # save_strategy="steps",
    # save_steps=1000,
    auto_find_batch_size=True,
    # predict_with_generate=True,
    # prediction_loss_only=False,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    fp16=True,
    group_by_length=True,
)

# For debugging
# train_ds = train_ds.select(range(100))
training_args = Seq2SeqTrainingArguments(
    output_dir="model_try_again",
    learning_rate=5e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=100,
    weight_decay=0.01,
    save_strategy="epoch",
    evaluation_strategy="steps",
    eval_steps=1,
    # save_strategy="steps",
    # save_steps=1000,
    auto_find_batch_size=True,
    # predict_with_generate=True,
    # prediction_loss_only=False,
    # load_best_model_at_end=True,
    remove_unused_columns=False,
    fp16=True,
    group_by_length=True,
)

# Do a single step and then stop
train_ds = train_ds.select(range(100))
training_args = Seq2SeqTrainingArguments(
    output_dir="model_first_try/draft_cs-en",
    learning_rate=5e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    max_steps=1,
    weight_decay=0.01,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    # eval_steps=1,
    # save_strategy="steps",
    # save_steps=1000,
    auto_find_batch_size=True,
    # predict_with_generate=True,
    # prediction_loss_only=False,
    # load_best_model_at_end=True,
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
    # compute_metrics=compute_metrics
)
trainer.train(resume_from_checkpoint=True)

import safetensors
new_model2 = MarianMTModel.from_pretrained("model_first_try/draft_cs-en/checkpoint-468780", use_safetensors=True)
state_dict = safetensors.torch.load_file("model_first_try/draft_cs-en/checkpoint-468780" + "/model.safetensors", device="cpu")
import pdb; pdb.set_trace()

inputs = tokenizer.batch_encode_plus([dataset["train"][2]["translation"]["cs"]], padding=True, return_tensors="pt").to(device)

# timer_start = time.perf_counter()
encoder_out = model.get_encoder()(**inputs)
# encode_time = time.perf_counter() - timer_start

# timer_start = time.perf_counter()
result = new_model.generate(
    encoder_outputs=encoder_out.copy(),
    generation_config=SAMPLE_GENERATION_CONFIG,
    renormalize_logits=True,
    output_scores=True,
    return_dict_in_generate=True)
# generate_time = time.perf_counter() - timer_start

# # torch.stack(result.scores, 1).gather(-1, result.sequences[:, 1:].unsqueeze(-1)).squeeze()[0]
# scores = model.compute_transition_scores(result.sequences, result.scores)

# timer_start = time.perf_counter()
# model(**inputs, decoder_input_ids=result.sequences[:1])
# encoder_out.last_hidden_state = encoder_out.last_hidden_state.repeat_interleave(NUM_SAMPLES, 0)
# result2 = model(
#     encoder_outputs=encoder_out,
#     attention_mask=inputs.attention_mask.repeat_interleave(NUM_SAMPLES, 0),
#     decoder_input_ids=result.sequences)

# scores2 = result2.logits.log_softmax(-1).gather(-1, result.sequences[:, 1:].unsqueeze(-1)).squeeze()

# score_time = time.perf_counter() - timer_start
texts = tokenizer.batch_decode(result.sequences, skip_special_tokens=True)

# import pdb; pdb.set_trace()
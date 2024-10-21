# Reference for how to generate and score with MarianMT
import time
import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig, MarianMTModel
from datasets import load_dataset

MAX_GENERATION_LENGTH = 256
NUM_SAMPLES = 512
SAMPLE_GENERATION_CONFIG = GenerationConfig(
    max_length=MAX_GENERATION_LENGTH,
    num_return_sequences=NUM_SAMPLES,
    do_sample=True,
    top_k=0,
    epsilon_cutoff=0.02
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-cs-en").to(device)
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-cs-en")


dataset = load_dataset("Helsinki-NLP/opus-100", "cs-en")

inputs = tokenizer.batch_encode_plus([dataset["train"][1]["translation"]["cs"]], padding=True, return_tensors="pt").to(device)

timer_start = time.perf_counter()
encoder_out = model.get_encoder()(**inputs)
encode_time = time.perf_counter() - timer_start

timer_start = time.perf_counter()
result = model.generate(
    encoder_outputs=encoder_out.copy(),
    generation_config=SAMPLE_GENERATION_CONFIG,
    renormalize_logits=True,
    output_scores=True,
    return_dict_in_generate=True)
generate_time = time.perf_counter() - timer_start

# torch.stack(result.scores, 1).gather(-1, result.sequences[:, 1:].unsqueeze(-1)).squeeze()[0]
scores = model.compute_transition_scores(result.sequences, result.scores)

timer_start = time.perf_counter()
model(**inputs, decoder_input_ids=result.sequences[:1])
encoder_out.last_hidden_state = encoder_out.last_hidden_state.repeat_interleave(NUM_SAMPLES, 0)
result2 = model(
    encoder_outputs=encoder_out,
    attention_mask=inputs.attention_mask.repeat_interleave(NUM_SAMPLES, 0),
    decoder_input_ids=result.sequences)

scores2 = result2.logits.log_softmax(-1).gather(-1, result.sequences[:, 1:].unsqueeze(-1)).squeeze()

score_time = time.perf_counter() - timer_start
texts = tokenizer.batch_decode(result.sequences, skip_special_tokens=True)

import pdb; pdb.set_trace()
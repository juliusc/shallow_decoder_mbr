import argparse
import time

import comet
import safetensors
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, MarianMTModel, GenerationConfig
import time

import torch
from transformers import GenerationConfig, AutoModel, AutoTokenizer, AutoConfig, M2M100ForConditionalGeneration

SRC_LANG_CODE = "ces_Latn"
TGT_LANG_CODE = "eng_Latn"

MAX_GENERATION_LENGTH = 256
NUM_SAMPLES = 64
SAMPLE_GENERATION_CONFIG = GenerationConfig(
    max_length=MAX_GENERATION_LENGTH,
    num_return_sequences=NUM_SAMPLES,
    do_sample=True,
    top_k=0,
    epsilon_cutoff=0.02
)

validation_ds = load_dataset("Helsinki-NLP/opus-100", name="cs-en", split="validation")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-3.3B").to(device).eval()
base_config = AutoConfig.from_pretrained("facebook/nllb-200-3.3B")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=SRC_LANG_CODE)

draft_model = M2M100ForConditionalGeneration.from_pretrained(
    "facebook/nllb-200-distilled-600M", torch_dtype=torch.float16).to(device).eval()
draft_config = AutoConfig.from_pretrained("facebook/nllb-200-distilled-600M")

tgt_lang_id = tokenizer.lang_code_to_id[TGT_LANG_CODE]



def get_sample_logprobs(output):
    scores_per_step = []
    for t in range(1, output.sequences.shape[1]):
        scores_per_step.append(output.scores[t-1].log_softmax(dim=-1).gather(1, output.sequences[:, t:t+1]))
    return torch.cat(scores_per_step, axis=1)


with torch.no_grad():
    for i in range(len(validation_ds)):
        inputs = tokenizer.batch_encode_plus([validation_ds[i]["translation"]["cs"]], padding=True, return_tensors="pt").to(device)

        timer_start = time.perf_counter()
        results_base = base_model.generate(**inputs, generation_config=SAMPLE_GENERATION_CONFIG, forced_bos_token_id=tgt_lang_id, output_scores=True, output_hidden_states=True, return_dict_in_generate=True)
        print(time.perf_counter() - timer_start)

        texts_base = tokenizer.batch_decode(results_base.sequences, skip_special_tokens=True)

        timer_start = time.perf_counter()
        results_draft = draft_model.generate(**inputs, generation_config=SAMPLE_GENERATION_CONFIG, forced_bos_token_id=tgt_lang_id, output_scores=True, return_dict_in_generate=True)
        print(time.perf_counter() - timer_start)

        texts_draft = tokenizer.batch_decode(results_draft.sequences, skip_special_tokens=True)

        draft_input = tokenizer.batch_encode_plus(texts_draft, padding=True, return_tensors="pt").to(device)

        timer_start = time.perf_counter()
        encoder_out = base_model.model.encoder(**inputs)
        last_hidden_state = encoder_out["last_hidden_state"]
        encoder_out["last_hidden_state"] = last_hidden_state.repeat_interleave(len(texts_draft),dim=0).to(device)

        decoder_inputs = tokenizer.batch_encode_plus([tgt for tgt in texts_draft], return_tensors="pt", padding=True)
        decoder_inputs["input_ids"][:, 0] = tgt_lang_id
        model_inputs = {}

        model_inputs["encoder_outputs"] = encoder_out
        model_inputs["attention_mask"] = inputs["attention_mask"].repeat_interleave(len(texts_draft),dim=0).to(device)
        model_inputs["labels"] = decoder_inputs["input_ids"].to(device)
        result = base_model(**model_inputs, output_hidden_states=True)
        y = result.logits.log_softmax(dim=-1).gather(2, model_inputs["labels"].unsqueeze(-1))
        print(time.perf_counter() - timer_start)




        import pdb; pdb.set_trace()
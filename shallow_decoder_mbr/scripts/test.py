import time

import torch
from transformers import GenerationConfig, AutoModel, AutoTokenizer, AutoConfig, M2M100ForConditionalGeneration

MAX_GENERATION_LENGTH = 256
NUM_SAMPLES = 128
NUM_SAMPLES = 1
SAMPLE_GENERATION_CONFIG = GenerationConfig(
    max_length=MAX_GENERATION_LENGTH,
    num_return_sequences=NUM_SAMPLES,
    do_sample=True,
    top_k=0
)
SRC_LANG_CODE = "ces_Latn"
TGT_LANG_CODE = "eng_Latn"

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
config = AutoConfig.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=SRC_LANG_CODE)
tgt_lang_id = tokenizer.lang_code_to_id[TGT_LANG_CODE]


with open("/home/jncc3/shallow_decoder_mbr/NLLB.cs-en.cs") as file:
    with torch.no_grad():
        for line in file:

            inputs = tokenizer.batch_encode_plus([line], padding=True, return_tensors="pt").to(device)

            timer_start = time.perf_counter()
            encoder_out = model.model.encoder(**inputs)
            encode_time = time.perf_counter() - timer_start

            timer_start = time.perf_counter()
            # Make sure to call encoder_out.copy() because it gets mutated by generate()
            result = model.generate(
                encoder_outputs=encoder_out.copy(),
                forced_bos_token_id=tgt_lang_id,
                generation_config=SAMPLE_GENERATION_CONFIG,
                renormalize_logits=True,
                output_scores=True,
                return_dict_in_generate=True)
            generate_time = time.perf_counter() - timer_start

            # torch.stack(result.scores, 1).gather(-1, result.sequences[:, 1:].unsqueeze(-1)).squeeze()[0]
            scores = model.compute_transition_scores(result.sequences, result.scores)

            timer_start = time.perf_counter()
            model.model(**inputs, decoder_input_ids=result.sequences[:1])
            encoder_out.last_hidden_state = encoder_out.last_hidden_state.repeat_interleave(NUM_SAMPLES, 0)
            result2 = model(
                encoder_outputs=encoder_out,
                attention_mask=inputs.attention_mask.repeat_interleave(NUM_SAMPLES, 0),
                decoder_input_ids=result.sequences)

            scores2 = result2.logits.log_softmax(-1).gather(-1, result.sequences[:, 1:].unsqueeze(-1)).squeeze()

            score_time = time.perf_counter() - timer_start

            # texts = tokenizer.batch_decode(result, skip_special_tokens=True)
            import pdb; pdb.set_trace()
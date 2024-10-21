import argparse
import time

import comet
import safetensors
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, MarianMTModel, GenerationConfig

MAX_GENERATION_LENGTH = 256
NUM_SAMPLES = 256
SAMPLE_GENERATION_CONFIG = GenerationConfig(
    max_length=MAX_GENERATION_LENGTH,
    num_return_sequences=NUM_SAMPLES,
    do_sample=True,
    top_k=0,
    epsilon_cutoff=0.02
)
SAMPLE_GENERATION_CONFIG2 = GenerationConfig(
    max_length=MAX_GENERATION_LENGTH,
    num_return_sequences=NUM_SAMPLES,
    do_sample=True,
    top_k=0,
    epsilon_cutoff=0.02
)
# SAMPLE_GENERATION_CONFIG = GenerationConfig(
#     max_length=MAX_GENERATION_LENGTH,
#     num_return_sequences=1,
#     num_beams=5
# )


def get_sample_logprobs(output):
    scores_per_step = []
    for t in range(1, output.sequences.shape[1]):
        scores_per_step.append(output.scores[t-1].log_softmax(dim=-1).gather(1, output.sequences[:, t:t+1]))


def main(args):
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-cs-en").to(device).eval()
    new_model = MarianMTModel.from_pretrained(args.draft_model_dir, use_safetensors=True)
    state_dict = safetensors.torch.load_file(args.draft_model_dir + "/model.safetensors", device="cpu")
    new_model.load_state_dict(state_dict, False)
    # TODO: This is due to the messed up training where decoder tokens were being tuned
    # when not intended
    # new_model.lm_head.weight = torch.nn.Parameter(state_dict["model.shared.weight"])
    new_model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-cs-en")

    new_model2 = MarianMTModel.from_pretrained("models/base_cs-en/checkpoint-375000", use_safetensors=True).eval().to(device)

    # from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir="model_first_try/draft_cs-en",
    #     learning_rate=5e-5,
    #     per_device_train_batch_size=128,
    #     per_device_eval_batch_size=128,
    #     max_steps=1,
    #     weight_decay=0.01,
    #     save_strategy="epoch",
    #     evaluation_strategy="epoch",
    #     # eval_steps=1,
    #     # save_strategy="steps",
    #     # save_steps=1000,
    #     auto_find_batch_size=True,
    #     # predict_with_generate=True,
    #     # prediction_loss_only=False,
    #     # load_best_model_at_end=True,
    #     remove_unused_columns=False,
    #     fp16=True,
    #     group_by_length=True,
    # )
    # trainer = Seq2SeqTrainer(
    #     model=new_model,
    #     args=training_args,
    # )
    # trainer.train(resume_from_checkpoint=True)


    validation_ds = load_dataset("Helsinki-NLP/opus-100", name="cs-en", split="validation")

    # from transformers import AutoConfig
    # config = AutoConfig.from_pretrained("Helsinki-NLP/opus-mt-cs-en")
    # new_model_config_dict = config.to_dict()
    # new_model_config_dict["decoder_layers"] = 1
    # new_model_config = type(config).from_dict(new_model_config_dict)
    # new_model = MarianMTModel(new_model_config).to(device)
    # new_model.model.encoder = model.model.encoder
    # # new_model.model.decoder.layers[0] = model.model.decoder.layers[0]
    # new_model.model.decoder.embed_tokens = model.model.decoder.embed_tokens


    with torch.no_grad():
        for i in range(1000):
            # TODO (julius) Figure out how to set the generation_config stuff automatically:
            new_model.generation_config = model.generation_config

            inputs = tokenizer.batch_encode_plus([validation_ds[i]["translation"]["cs"]], padding=True, return_tensors="pt").to(device)

            timer_start = time.perf_counter()
            encoder_out = model.model.encoder(**inputs)
            print(time.perf_counter() - timer_start)

            timer_start = time.perf_counter()
            results = model.generate(encoder_outputs=encoder_out.copy(), generation_config=SAMPLE_GENERATION_CONFIG, output_scores=True, return_dict_in_generate=True)
            print(time.perf_counter() - timer_start)

            texts = tokenizer.batch_decode(results.sequences, skip_special_tokens=True)

            timer_start = time.perf_counter()
            encoder_out = new_model.model.encoder(**inputs)
            print(time.perf_counter() - timer_start)

            timer_start = time.perf_counter()
            results2 = new_model.generate(encoder_outputs=encoder_out.copy(), generation_config=SAMPLE_GENERATION_CONFIG2, output_scores=True, return_dict_in_generate=True)
            print(time.perf_counter() - timer_start)

            texts2 = tokenizer.batch_decode(results2.sequences, skip_special_tokens=True)

            results3 = new_model2.generate(**inputs, generation_config=SAMPLE_GENERATION_CONFIG, output_scores=True, return_dict_in_generate=True)
            texts3 = tokenizer.batch_decode(results3.sequences, skip_special_tokens=True)

            import pdb; pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_name", type=str, help="HuggingFace name of the main model.")

    parser.add_argument(
        "draft_model_dir", type=str, help="Data split. Either 'dev' or 'test'.")

    parser.add_argument(
        "--comet_repo", help="Huggingface COMET model name. Must pass --comet_repo or --comet_path")

    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    main(args)

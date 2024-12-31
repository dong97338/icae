import json
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
from peft import LoraConfig
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from safetensors.torch import load_file
import argparse
import os
from pipeline_config import PipelineConfig

# Set the computation device
device = "cuda"

# Parse stage argument first
stage_parser = argparse.ArgumentParser()
stage_parser.add_argument('--stage', type=int, choices=[1, 2, 3], required=True,
                          help='Select specific inference stage (1, 2, or 3).')
stage_args, remaining_args = stage_parser.parse_known_args()

# Now parse model, data, training, and pipeline arguments from remaining args
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, PipelineConfig))
model_args, data_args, training_args, pipeline_config = parser.parse_args_into_dataclasses(args=remaining_args)

# Set training_args.use_r_tokens based on the stage
if stage_args.stage == 1:
    training_args.use_r_tokens = False
    checkpoint_path = pipeline_config.q_only_checkpoint
elif stage_args.stage == 2:
    training_args.use_r_tokens = True
    checkpoint_path = pipeline_config.q_r_checkpoint
elif stage_args.stage == 3:
    training_args.use_r_tokens = True
    checkpoint_path = pipeline_config.q_r_checkpoint
else:
    raise ValueError("Invalid stage selected. Choose from 1, 2, or 3.")

# Define Lora configuration
lora_config = LoraConfig(
    r=model_args.lora_r,
    lora_alpha=32,
    lora_dropout=model_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

# Initialize model and send it to CUDA device
model = ICAE(model_args, training_args, lora_config)
model = model.to(device)

# Load the fine-tuned checkpoint
print(f"Loading trained checkpoint from {checkpoint_path}")
state_dict = load_file(os.path.join(checkpoint_path, "model.safetensors"))
model.load_state_dict(state_dict, strict=False)  # only load LoRA and memory token embeddings

# Set the model to evaluation mode
model.eval()

# Load the input data
file_path = "./test.jsonl"
with open(file_path, "r") as f:
    lines = f.readlines()

# Prepare the model for evaluation
with torch.no_grad():
    with open("inference_output_stage_{}.jsonl".format(stage_args.stage), "w") as f:
        for line in tqdm(lines):
            data = json.loads(line.strip())
            input_text = data.get('input', '')
            if not input_text:
                continue

            # Tokenize input text
            tokenized_text = model.tokenizer(
                input_text,
                truncation=True,
                max_length=5120,
                padding=False,
                return_attention_mask=False
            )
            # Generate compressed outputs
            input_ids = torch.LongTensor([tokenized_text['input_ids']]).to(device)
            memory_slots = model._compress(input_ids)

            # Prepare the decoder input
            # 모든 메모리 토큰을 포함하도록 수정
            mem_tokens = list(range(model.vocab_size, model.vocab_size + model.total_mem_size))
            prompt_ids = mem_tokens + [model.ae_token_id]
            prompt_ids = torch.LongTensor([prompt_ids]).to(device)
            prompt_answer_embs = model.tokens_to_embeddings(prompt_ids)
            memory_slots = memory_slots.to(prompt_answer_embs)

            # Concatenate memory slots and prompt embeddings
            decoder_input_embeddings = torch.cat((memory_slots, prompt_answer_embs), dim=1)
            output = decoder_input_embeddings.clone()

            generate_text = []
            past_key_values = None

            # Generate text output
            for _ in range(512):
                with model.icae.disable_adapter():
                    out = model.icae(
                        inputs_embeds=output,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                logits = out.logits[:, -1, :model.vocab_size - 1]
                past_key_values = out.past_key_values

                next_token_id = torch.argmax(logits, dim=-1)

                if next_token_id.item() == model.eos_id:  # EOS token
                    break

                output = model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)
                generate_text.append(next_token_id.item())

            generated_text = model.tokenizer.decode(
                generate_text,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Structure output data
            output_ = {
                "input": input_text,
                "output": generated_text
            }

            f.write(json.dumps(output_, ensure_ascii=False) + "\n")

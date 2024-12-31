import json, torch
from tqdm import tqdm
from transformers import HfArgumentParser
from peft import LoraConfig
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from safetensors.torch import load_file
from pipeline_config import PipelineConfig

device = "cuda"
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
lora_config = LoraConfig(r=512, lora_alpha=32, lora_dropout=model_args.lora_dropout, bias="none", task_type="CAUSAL_LM")
model = ICAE(model_args, training_args, lora_config)
print(f"Loading trained checkpoint from {training_args.output_dir}")
state_dict = load_file(training_args.output_dir)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)

## ck
# data_path = PipelineConfig.target_q_to_eng_q_path

with open("./cc_ko/stage1_eval.jsonl","r") as f: lines = f.readlines()
model.eval()
with torch.no_grad():
    with open("stage1_eval_512r.out","w", encoding="utf-8") as f:
        for line in tqdm(lines[:10]):
            data = json.loads(line.strip())
            tok = model.tokenizer(data["input"], truncation=True, max_length=1024, padding="max_length", return_attention_mask=False)
            inp = torch.LongTensor([tok['input_ids']]).to(device)
            mem = model._compress2(inp)
            pm = torch.LongTensor([[model.ae_token_id]]).to(device)
            pe = model.tokens_to_embeddings(pm)
            mem = mem.to(pe)
            out = torch.cat((mem.unsqueeze(0), pe), dim=1).clone()
            gen, pkv = [], None
            for _ in range(512):
                with model.icae.disable_adapter():
                    r = model.icae(inputs_embeds=out, past_key_values=pkv, use_cache=True)
                l = r.logits[:,-1,:model.vocab_size-1]
                pkv = r.past_key_values
                nt = torch.argmax(l, dim=-1)
                if nt.item() == 2: break
                out = model.icae.get_base_model().model.embed_tokens(nt).unsqueeze(1).to(device)
                gen.append(nt.item())
            txt = model.tokenizer.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            f.write(json.dumps({"input": data["input"], "output": txt}, ensure_ascii=False) + "\n")
            print(model.training_args.use_r_tokens, model.q_mem_size, model.r_mem_size, model.total_mem_size)

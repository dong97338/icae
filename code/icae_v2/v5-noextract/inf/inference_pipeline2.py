# inference_stage1_autoregressive.py

import json
import torch
import os
from tqdm import tqdm
from datasets import load_dataset

# 필요한 클래스/함수/설정 로드
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from pipeline_config import PipelineConfig
from peft import LoraConfig
from safetensors.torch import load_file
from training_utils import pretrain_tokenize_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_stage1_inference_autoregressive():
    """
    Stage1 방식(Q 메모리 사용)으로 학습된 모델을, 
    실제 인퍼런스(오토리그레시브 생성) 방식으로 추론 수행하는 예시 코드.
    """
    # -------------------------------------------------------------
    # 1) 필요한 인자들 셋업 (ModelArguments, DataArguments, TrainingArguments, PipelineConfig)
    # -------------------------------------------------------------
    model_args = ModelArguments(
        model_name_or_path="mistralai/Mistral-7B-v0.1",
        train=False,   # 인퍼런스 용
        lora_r=128,
        lora_dropout=0.05
    )
    data_args = DataArguments(
        data_path=None, 
        debug_data=False
    )
    training_args = TrainingArguments(
        output_dir="./out_inference_stage1",
        per_device_eval_batch_size=1,
        num_train_epochs=1,  # dummy
        max_steps=1,         # dummy
        bf16=True,
        fp16=False,
        do_train=False,
        do_eval=False,
        logging_steps=10,
        evaluation_strategy="no",
        save_strategy="no",
        overwrite_output_dir=True,
        model_max_length=1024,  
        fixed_q_mem_size=256,  # Stage1에서 사용하는 Q 메모리 크기
        fixed_r_mem_size=0,    # R 토큰은 사용하지 않음
        use_r_tokens=False,    # Stage1 -> R 토큰 불필요
        lm_ratio=0.0,
    )
    pipeline_config = PipelineConfig()
    # 필요한 데이터 경로(예: stage1_train.jsonl, stage1_eval.jsonl 등)는 pipeline_config.* 에서 불러올 수도 있음
    # 여기서는 pipeline_config.target_q_to_eng_q_path를 예시로 사용

    # -------------------------------------------------------------
    # 2) 로라 Config 준비 & 모델 로드
    # -------------------------------------------------------------
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print("[+] Loading ICAE model (stage1: Q-only, autoregressive decoding)...")
    model = ICAE(model_args, training_args, lora_config).to(device)

    # Stage1 훈련 완료 후 저장된 safetensors 경로
    q_only_checkpoint = "/ssd1/donghyeon/icae/out/1210-mintest"  # 예시
    state_dict_path = os.path.join(q_only_checkpoint, "model.safetensors")
    if os.path.exists(state_dict_path):
        print(f"[+] Loading state_dict from {state_dict_path}")
        state_dict = load_file(state_dict_path)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        print("[!] Warning: No checkpoint found. Using base model weights only.")
    
    # -------------------------------------------------------------
    # 3) 평가/테스트용 데이터 로드
    # -------------------------------------------------------------
    data_path = pipeline_config.target_q_to_eng_q_path  # stage1용 데이터 (Q -> eng_q) 예시
    if not os.path.exists(data_path):
        raise ValueError(f"No data at {data_path}")
    
    dataset = load_dataset("json", data_files={"test": data_path})["test"]
    # 예시로 3개만 추론
    dataset = dataset.select(range(min(3, len(dataset))))

    # -------------------------------------------------------------
    # 4) 오토리그레시브 생성 함수
    #    (입력 -> _compress() -> memory_slots -> decoder에 AE 토큰 붙인 뒤, step-by-step argmax 생성)
    # -------------------------------------------------------------
    def autoregressive_inference_single_sample(input_text: str, max_new_tokens=128):
        """
        (수정판) bfloat16 mismatch 에러를 피하기 위해,
        memory_slots와 임베딩 텐서를 모델의 임베딩 dtype으로 통일해준다.
        """
        # 0) 모델 임베딩 dtype 확인
        #    (Mistral + LoRA 등에서, 보통 torch.bfloat16 이 설정되어 있음)
        embed_dtype = model.icae.get_base_model().model.embed_tokens.weight.dtype
        # 또는 model.icae.get_input_embeddings().weight.dtype 로도 확인 가능

        # 1) 입력을 tokenizer
        tokenized_text = model.tokenizer(
            input_text,
            truncation=True,
            max_length=training_args.model_max_length,
            padding=False,
            return_attention_mask=False
        )
        input_ids = torch.LongTensor([tokenized_text['input_ids']]).to(device)

        # 2) encoder 쪽 압축
        memory_slots = model._compress(input_ids)  
        # _compress() 결과도 float32로 나올 가능성이 있으므로, 다음과 같이 캐스팅
        memory_slots = memory_slots.to(device, dtype=embed_dtype) 
        # shape: (1, q_mem_size, hidden_dim)

        # 3) 디코더 입력용 초기 임베딩 구성
        #    (a) AE 토큰 ID를 텐서로 만들고 임베딩
        initial_ids = torch.LongTensor([[model.ae_token_id]]).to(device)
        initial_embs = model.tokens_to_embeddings(initial_ids)  # float32일 수 있음
        # bfloat16으로 변환
        initial_embs = initial_embs.to(device, dtype=embed_dtype)

        #    (b) memory_slots + AE 토큰 임베딩을 concat
        decoder_input_embs = torch.cat([memory_slots, initial_embs], dim=1)  # (1, q_mem_size+1, dim)

        # 4) 오토리그레시브 생성
        output_tokens = []
        past_key_values = None
        current_embs = decoder_input_embs  # 처음에는 memory_slots + AE 임베딩

        for step in range(max_new_tokens):
            # decoder forward
            with torch.no_grad(), model.icae.disable_adapter():
                out = model.icae(
                    inputs_embeds=current_embs,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            logits = out.logits[:, -1, :model.vocab_size-1]
            past_key_values = out.past_key_values

            next_token_id = torch.argmax(logits, dim=-1)  # shape (1,)

            # EOS check
            if next_token_id.item() == model.eos_id:
                break

            # 새로 생성된 토큰 임베딩
            # 여기서 next_token_id가 int->LongTensor이므로, 임베딩 dtype도 맞춰야 함
            new_token_emb = model.icae.get_base_model().model.embed_tokens(next_token_id)
            new_token_emb = new_token_emb.to(dtype=embed_dtype)  # bfloat16으로
            new_token_emb = new_token_emb.unsqueeze(1)  # (1, 1, dim)
            current_embs = new_token_emb  # 다음 step 입력
            output_tokens.append(next_token_id.item())
        
        # 5) 완성된 output_tokens를 디코딩
        generated_text = model.tokenizer.decode(
            output_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return generated_text


    # -------------------------------------------------------------
    # 5) 실제로 inference 수행 & 저장
    # -------------------------------------------------------------
    results = []
    for sample in tqdm(dataset, desc="Autoregressive Inference"):
        input_text = sample["input"]
        output_label = sample["output"]
        
        # 오토리그레시브 생성
        generated_text = autoregressive_inference_single_sample(input_text)

        results.append({
            "input": input_text,
            "output_label": output_label,
            "model_generated": generated_text
        })

    # 결과 저장
    output_file = "stage1_inference_autoregressive_result.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[+] Done. Results saved at {output_file}")

if __name__ == "__main__":
    run_stage1_inference_autoregressive()

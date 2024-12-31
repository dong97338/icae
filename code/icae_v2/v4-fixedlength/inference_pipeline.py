# inference_stage1.py
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
import os

# 필요한 함수/클래스 로드
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from pipeline_config import PipelineConfig
from training_utils import pretrain_tokenize_function, DataCollatorForDynamicPadding
from peft import LoraConfig
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def single_pass_inference_and_decode(model, batch, data_collator):
    """
    단일 배치에 대해:
      1) model.forward()로 logits 계산
      2) logits에서 teacher forcing 식으로 전체 시퀀스를 argmax 디코딩
      3) 디코딩된 텍스트 반환
    """
    # data_collator를 이용해 동적 패딩
    collated = data_collator(batch)
    # collated = {
    #   'input_ids': [batch_size, seq_len], 
    #   'labels': [batch_size, seq_len], 
    #   'prompt_answer_ids': [batch_size, seq_len]
    # }
    for k, v in collated.items():
        collated[k] = v.to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=collated["input_ids"],
            labels=collated["labels"],
            prompt_answer_ids=collated["prompt_answer_ids"]
        )
        logits = outputs["logits"]  # (batch_size, seq_len, vocab_size)
    
    # teacher forcing 식으로 -> logits[:, :, :].argmax(dim=-1)
    # batch 별로 하나씩 디코딩
    batch_decoded = []
    pred_ids = logits.argmax(dim=-1)  # (batch_size, seq_len)
    for i in range(pred_ids.size(0)):
        # prompt_answer_ids[i] 와 동일한 길이를 사용
        seq_length = collated["prompt_answer_ids"][i].size(0)
        # 인퍼런스 결과(=argmax 토큰) 잘라서 디코딩
        tokens = pred_ids[i][:seq_length]
        text = model.tokenizer.decode(tokens, skip_special_tokens=False)
        batch_decoded.append(text)
    return batch_decoded

def run_stage1_inference():
    """
    1) Stage 1 때처럼 Q 토큰만 사용하는 모델을 로드
    2) stage1용 pretrain_tokenize_function으로 토큰화
    3) single_pass_inference_and_decode()로 한 번에 logits -> argmax 디코딩
    4) 결과를 출력
    """
    # -------------------------------------------------------------
    # 1) 필요한 인자들 셋업 (ModelArguments, DataArguments, TrainingArguments, PipelineConfig)
    #    여기서는 HfArgumentParser 없이, 예시로 직접 생성
    # -------------------------------------------------------------
    model_args = ModelArguments(
        model_name_or_path="mistralai/Mistral-7B-v0.1",
        train=False,       # 인퍼런스 시에는 굳이 train=True 아니어도 됨
        lora_r=128,
        lora_dropout=0.05
    )
    data_args = DataArguments(data_path=None, debug_data=False)
    training_args = TrainingArguments(
        output_dir="./out_inference_stage1",
        per_device_train_batch_size=1,
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
        model_max_length=1024,   # 예시
        fixed_q_mem_size=256,
        fixed_r_mem_size=0,      # Stage1은 R 토큰 안씀: 이것도 0으로 해야 오류 안생김
        use_r_tokens=False,      # Stage1은 R 토큰 안씀
        lm_ratio=0.0,
    )
    pipeline_config = PipelineConfig()

    # -------------------------------------------------------------
    # 2) LoRA Config 준비 & 모델 로드
    # -------------------------------------------------------------
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    print("[+] Loading ICAE model (stage1: Q-only)...")
    model = ICAE(model_args, training_args, lora_config).to(device)
    
    # LoRA + token embedding만 safetensors에 저장되어 있다고 가정
    # (train_pipeline에서 stage1 완료 후 저장된 checkpoint)
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
    # 3) 평가용(혹은 예시용) 데이터 불러오기
    #    여기서는 pipeline_config.target_q_to_eng_q_path (stage1용) 사용 예시
    # -------------------------------------------------------------
    data_path = pipeline_config.target_q_to_eng_q_path
    if not os.path.exists(data_path):
        raise ValueError(f"No data at {data_path}")
    
    dataset = load_dataset("json", data_files={"test": data_path})["test"]
    
    # -------------------------------------------------------------
    # 4) dataset 일부만 뽑아 토큰화 -> single_pass_inference_and_decode
    # -------------------------------------------------------------
    # MEM_TOKENS: stage1은 Q 토큰만 -> fixed_q_mem_size 개
    #             vocab_size 기준 offset 계산은 model.vocab_size 등이 관여
    memory_size = training_args.fixed_q_mem_size
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))
    
    # pretrain_tokenize_function 재활용
    def preprocess_fn(examples):
        return pretrain_tokenize_function(
            examples,
            model=model,
            mem=MEM_TOKENS,
            lm_ratio=training_args.lm_ratio  # 0.0
        )
    
    print("[+] Tokenizing dataset with pretrain_tokenize_function (Stage1 style)")
    tokenized_dataset = dataset.map(
        preprocess_fn,
        batched=True,
        batch_size=training_args.per_device_eval_batch_size
    )
    tokenized_dataset = tokenized_dataset.select(range(3))
    # 추론용 데이터 콜레이터
    data_collator = DataCollatorForDynamicPadding(model.pad_token_id)
    
    # -------------------------------------------------------------
    # 5) 실제로 single_pass_inference_and_decode 실행
    # -------------------------------------------------------------
    results = []
    for sample in tqdm(tokenized_dataset, desc="Inference"):
        batch = [sample]  # collator에 넣으려면 list 형태여야 함
        decoded_texts = single_pass_inference_and_decode(model, batch, data_collator)
        results.append({
            "input": sample["input"],
            "output_label": sample["output"],
            "model_decoded": decoded_texts[0]
        })
    
    # -------------------------------------------------------------
    # 6) 결과 확인 (JSONLines 등으로 저장 예시)
    # -------------------------------------------------------------
    output_file = "stage1_inference_result.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"[+] Done. Results saved at {output_file}")

if __name__ == "__main__":
    run_stage1_inference()

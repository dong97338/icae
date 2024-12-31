# train_pipeline.py
import torch
from transformers import HfArgumentParser
from peft import LoraConfig
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from pipeline_config import PipelineConfig
from datasets import load_dataset
import os
from training_utils import pretrain_tokenize_function, instruct_ft_tokenize_function
from training_utils import DataCollatorForDynamicPadding, train_model
from safetensors.torch import load_file, save_file
import inspect
import argparse

def train_pipeline():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, PipelineConfig))
    model_args, data_args, training_args, pipeline_config = parser.parse_args_into_dataclasses()
    
    # --- 공통 설정 ---
    training_args.lr_scheduler_type = "cosine"
    training_args.warmup_ratio = 0.05
    training_args.bf16 = True
    training_args.fp16 = False
    training_args.report_to = ["wandb"]
    training_args.remove_unused_columns = False
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    for stage_idx in range(3):
        if not pipeline_config.run_stages[stage_idx]:
            print(f"Skipping stage {stage_idx}...")
            continue
        
        # 1) 스테이지별 TrainingArguments 복사 & 세부 설정
        original_args_dict = vars(training_args)

        # TrainingArguments.__init__() 시그니처에서 valid 파라미터명만 추출
        valid_keys = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

        # 필요 없는 키(예: distributed_state)는 제외
        filtered_args = {k: v for k, v in original_args_dict.items() if k in valid_keys}

        stage_training_args = TrainingArguments(**filtered_args)
        stage_training_args.num_train_epochs = pipeline_config.epochs[stage_idx]
        
        # 예: stage_idx==0 → Q-only, stage_idx==1 → Q+R, stage_idx==2 → Q+R ...
        if stage_idx == 0:
            stage_training_args.use_r_tokens = False
        else:
            stage_training_args.use_r_tokens = True
        
        # 2) 스테이지별 데이터 경로
        train_file = pipeline_config.train_files[stage_idx]
        eval_file = pipeline_config.eval_files[stage_idx]

        # 3) 스테이지별 체크포인트 디렉토리
        current_ckpt_dir = pipeline_config.get_stage_checkpoint_dir(stage_idx)
        stage_training_args.output_dir = current_ckpt_dir
        os.makedirs(current_ckpt_dir, exist_ok=True)

        # 4) 이전 스테이지 체크포인트 경로
        if stage_idx > 0:
            prev_ckpt_dir = pipeline_config.get_stage_checkpoint_dir(stage_idx - 1)
            prev_safetensor_path = os.path.join(prev_ckpt_dir, "model.safetensors")
        else:
            prev_safetensor_path = None

        # 5) 모델 생성
        print(f"[Stage {stage_idx}] Initializing model...")
        model = ICAE(model_args, stage_training_args, lora_config).cuda()

        # 6) 이전 스테이지 체크포인트 로드 (stage_idx=0은 베이스 모델에서 시작)
        if prev_safetensor_path and os.path.exists(prev_safetensor_path):
            print(f"[Stage {stage_idx}] Loading checkpoint from {prev_safetensor_path}")
            state_dict = load_file(prev_safetensor_path)
            model.load_state_dict(state_dict, strict=False)
        else:
            if stage_idx > 0:
                print(f"[Warning] No checkpoint found at {prev_safetensor_path} (stage {stage_idx}). Using base weights.")
        
        # 7) 데이터 로드
        dataset = load_dataset("json", data_files={"train": train_file, "validation": eval_file})
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        # 8) 전처리(토큰화)
        memory_size = model.training_args.fixed_q_mem_size
        if stage_training_args.use_r_tokens:
            memory_size += model.training_args.fixed_r_mem_size
        
        MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

        if stage_idx == 0 or stage_idx == 1:
            # stage0/stage1 => JSON에 "input", "output"이 있다고 가정
            def preprocess_fn(examples):
                return pretrain_tokenize_function(
                    examples,
                    model=model,
                    mem=MEM_TOKENS,
                    lm_ratio=stage_training_args.lm_ratio
                )
        else:
            # stage2 => JSON에 "input", "prompt", "answer"가 있다고 가정
            def preprocess_fn(examples):
                return instruct_ft_tokenize_function(
                    examples,
                    model=model,
                    mem=MEM_TOKENS
                )
        train_dataset = train_dataset.map(
            preprocess_fn,
            batched=True,
            batch_size=stage_training_args.per_device_train_batch_size,
        )
        eval_dataset = eval_dataset.map(
            preprocess_fn,
            batched=True,
            batch_size=stage_training_args.per_device_eval_batch_size,
        )

        data_collator = DataCollatorForDynamicPadding(model.pad_token_id)

        # 9) 실제 학습
        trained_model = train_model(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=stage_training_args,
            data_collator=data_collator
        )

        print(f"[Stage {stage_idx}] Training done. Checkpoint is saved at {current_ckpt_dir}")

        # ---- (메모리 정리) ----
        # 학습된 모델(trained_model) 객체를 해제하여 GPU 메모리 반환
        del trained_model
        del model
        torch.cuda.empty_cache()

        # 다음 스테이지로 넘어가면, 위에서 저장된 (current_ckpt_dir의 model.safetensors)를 
        # 다시 로딩하는 절차를 거치게 됨

if __name__ == "__main__":
    train_pipeline()

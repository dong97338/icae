# train_pipeline.py
import torch
from transformers import HfArgumentParser
from peft import LoraConfig
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from pipeline_config import PipelineConfig
from datasets import load_dataset
import os
from training_utils import pretrain_tokenize_function, instruct_ft_tokenize_function, train_model  # DataCollatorForDynamicPadding 제거
from safetensors.torch import load_file, save_file
import inspect

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
        if pipeline_config.epochs[stage_idx] == 0:
            print(f"Skipping stage {stage_idx} (epochs=0)...")
            continue
        
        # 1) 스테이지별 TrainingArguments 복사 & 세부 설정
        # TrainingArguments에 있는 valid 키만 추출
        import inspect
        original_args_dict = vars(training_args)
        valid_keys = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        filtered_args = {k: v for k, v in original_args_dict.items() if k in valid_keys}
        stage_training_args = TrainingArguments(**filtered_args)
        stage_training_args.num_train_epochs = pipeline_config.epochs[stage_idx]
        
        if stage_idx == 0:
            stage_training_args.use_r_tokens = False
        else:
            stage_training_args.use_r_tokens = True
        ###############################################
        # (추가) group_by_length를 켜고, length_column_name 지정
        ###############################################
        stage_training_args.group_by_length = True
        stage_training_args.length_column_name = "label_length"
        
        train_file = pipeline_config.train_files[stage_idx]
        eval_file = pipeline_config.eval_files[stage_idx]

        stage_output_dir = os.path.join(training_args.output_dir, f"stage_{stage_idx}")
        stage_training_args.output_dir = stage_output_dir
        os.makedirs(stage_output_dir, exist_ok=True)

        if stage_idx > 0:
            prev_ckpt_dir = os.path.join(training_args.output_dir, f"stage_{stage_idx-1}")
            prev_safetensor_path = os.path.join(prev_ckpt_dir, "model.safetensors")
        else:
            prev_safetensor_path = None

        # 5) 모델 생성
        print(f"[Stage {stage_idx}] Initializing model...")
        model = ICAE(model_args, stage_training_args, lora_config).cuda()

        if prev_safetensor_path and os.path.exists(prev_safetensor_path):
            print(f"[Stage {stage_idx}] Loading checkpoint from {prev_safetensor_path}")
            state_dict = load_file(prev_safetensor_path)
            model.load_state_dict(state_dict, strict=False)
        else:
            if stage_idx > 0:
                print(f"[Warning] No checkpoint found at {prev_safetensor_path} (stage {stage_idx}). Using base weights.")
        model = torch.compile(model, mode="reduce-overhead", backend="inductor")

        # 7) 데이터 로드
        dataset = load_dataset("json", data_files={"train": train_file, "validation": eval_file})
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        # 8) 전처리(토큰화)
        memory_size = model.training_args.fixed_q_mem_size
        if stage_training_args.use_r_tokens:
            memory_size += model.training_args.fixed_r_mem_size
        
        MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

        if stage_idx == 0 or stage_idx == 1 or stage_idx == 2:
            # stage0/stage1 => JSON에 "input", "output"이 있다고 가정
            def preprocess_fn(examples):
                return pretrain_tokenize_function(
                    examples,
                    model=model,
                    mem=MEM_TOKENS,
                )
        # else:
        #     # stage2 => JSON에 "input", "prompt", "answer"가 있다고 가정
        #     def preprocess_fn(examples):
        #         return instruct_ft_tokenize_function(
        #             examples,
        #             model=model,
        #             mem=MEM_TOKENS
        #         )
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

        ###########################################
        # (추가) 라벨 길이(length) 컬럼을 만들어 주기
        ###########################################
        def add_label_length_column(examples):
            """
            labels에서 -100(IGNORE) 아닌 실제 토큰만 세어서 length를 구하기 위한 함수입니다.
            """
            lengths = []
            for label_ids in examples["labels"]:
                # label_ids 중 -100이 아닌 것(실제 토큰)의 개수를 구하기 위한 함수
                valid_label_count = sum(1 for token_id in label_ids if token_id != -100)
                lengths.append(valid_label_count)
            return {"label_length": lengths}

        train_dataset = train_dataset.map(add_label_length_column, batched=True)
        eval_dataset = eval_dataset.map(add_label_length_column, batched=True)

        # 9) 실제 학습 (data_collator 제거 → Trainer 기본 패딩/콜레이션 사용)
        trained_model = train_model(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=stage_training_args
        )

        print(f"[Stage {stage_idx}] Training done. Checkpoint is saved at {stage_output_dir}")

        del trained_model
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train_pipeline()

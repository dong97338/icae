import torch
from transformers import HfArgumentParser
from peft import LoraConfig
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from pipeline_config import PipelineConfig
from datasets import load_dataset
import os
from training_utils import pretrain_tokenize_function, instruct_ft_tokenize_function, DataCollatorForDynamicPadding, train_model
from copy import deepcopy
from transformers import Trainer
from safetensors.torch import load_file

def add_length_column(example):
    # prompt_answer_ids 길이를 length로 저장
    example["length"] = len(example["prompt_answer_ids"])
    return example

def train_stage(
    model_args,
    training_args,
    pipeline_config,
    tokenizer_func,   # pretrain_tokenize_function or instruct_ft_tokenize_function
    data_path_train,
    data_path_eval=None,
    num_epochs=3,
    use_r_tokens=False,
    prev_model=None,
    prev_dataset=None,
    ewc=False,
    ewc_lambda=0.5,
    load_checkpoint=None,
    lora_config=None,
    stage_name="Stage"
):
    """
    - 하나의 함수 안에서:
      1) TrainingArguments 복사 & 수정
      2) 모델 생성/로딩
      3) 데이터셋 로드 & 토크나이징 & length column 추가
      4) 버킷 샘플링 + Collator + Trainer 세팅
      5) 훈련 실행 + 체크포인트 저장
    - ewc: EWC 사용 여부
    """
    print(f"{stage_name}: Running training with use_r_tokens={use_r_tokens}")

    from copy import deepcopy
    import torch
    from datasets import load_dataset

    # 1) training args 복제
    stage_training_args = deepcopy(training_args)
    stage_training_args.num_train_epochs = num_epochs
    stage_training_args.use_r_tokens = use_r_tokens

    # 2) 모델 생성
    model = ICAE(model_args, stage_training_args, lora_config)
    model.to("cuda")
    model = torch.compile(model, mode="max-autotune")

    # (옵션) load checkpoint
    if load_checkpoint is not None and os.path.exists(load_checkpoint):
        from safetensors.torch import load_file
        state_dict = load_file(os.path.join(load_checkpoint, "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
    elif prev_model is not None:
        model.load_state_dict(prev_model.state_dict(), strict=False)

    # 3) 데이터셋 로드 & tokenize
    raw_dataset = load_dataset("json", data_files={"train": data_path_train})["train"]
    # tokenize
    memory_size = stage_training_args.fixed_q_mem_size + (stage_training_args.fixed_r_mem_size if use_r_tokens else 0)
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))
    train_dataset = raw_dataset.map(
        tokenizer_func,
        batched=True,
        batch_size=stage_training_args.per_device_train_batch_size,
        fn_kwargs={"model": model, "mem": MEM_TOKENS, "lm_ratio": stage_training_args.lm_ratio}
    )

    # eval dataset
    eval_dataset = None
    if data_path_eval:
        raw_eval_dataset = load_dataset("json", data_files={"validation": data_path_eval})["validation"]
        eval_dataset = raw_eval_dataset.map(
            tokenizer_func,
            batched=True,
            fn_kwargs={"model": model, "mem": MEM_TOKENS}
        )

    # length 컬럼 추가
    def add_length_column(example):
        example["length"] = len(example["prompt_answer_ids"])
        return example

    train_dataset = train_dataset.map(add_length_column)
    if eval_dataset:
        eval_dataset = eval_dataset.map(add_length_column)

    # EWC 설정
    if ewc and (prev_model is not None or load_checkpoint):
        # prev_dataset 필요
        from training_utils import EWC
        ewc_obj = EWC(prev_model if prev_model else model, prev_dataset if prev_dataset else train_dataset)
        original_loss_fn = model.loss_fct
        def new_loss_fn(pred, target):
            return original_loss_fn(pred, target) + ewc_lambda * ewc_obj.penalty(model)
        model.loss_fct = new_loss_fn

    # 4) 버킷 샘플링 + Collator + Trainer
    from training_utils import LengthGroupedBucketSampler, DataCollatorForBucketPadding
    from torch.utils.data import DataLoader
    from transformers import Trainer

    data_collator = DataCollatorForBucketPadding(model.pad_token_id)
    train_sampler = LengthGroupedBucketSampler(
        data_source=train_dataset,
        batch_size=stage_training_args.per_device_train_batch_size,
        lengths=train_dataset["length"],
        shuffle=True,
        grouping_factor=50
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=data_collator
    )

    eval_dataloader = None
    if eval_dataset:
        eval_sampler = LengthGroupedBucketSampler(
            data_source=eval_dataset,
            batch_size=stage_training_args.per_device_eval_batch_size,
            lengths=eval_dataset["length"],
            shuffle=False,
            grouping_factor=50
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_sampler=eval_sampler,
            collate_fn=data_collator
        )

    trainer = Trainer(
        model=model,
        args=stage_training_args,
        data_collator=data_collator
    )
    def custom_train_dataloader():
        return train_dataloader
    def custom_eval_dataloader():
        return eval_dataloader if eval_dataloader else None
    trainer.get_train_dataloader = custom_train_dataloader
    trainer.get_eval_dataloader = custom_eval_dataloader

    # 5) 학습 진행
    train_result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    if eval_dataloader:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return model, train_dataset


class EWC:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._fisher = {}
        self.compute_fisher()

    def compute_fisher(self):
        for n, p in self.params.items():
            self._means[n] = p.data.clone()
            self._fisher[n] = torch.zeros_like(p.data)
        
        self.model.train()
        for batch in self.dataset:
            self.model.zero_grad()
            output = self.model(**batch)
            loss = output["loss"]
            loss.backward()
            
            for n, p in self.params.items():
                self._fisher[n].data += p.grad.data ** 2 / len(self.dataset)
    
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._fisher:
                loss += (self._fisher[n] * (p - self._means[n]) ** 2).sum()
        return loss

def train_pipeline():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, PipelineConfig))
    model_args, data_args, training_args, pipeline_config = parser.parse_args_into_dataclasses()

    training_args.report_to = ["wandb"]
    training_args.remove_unused_columns = False

    import argparse
    stage_parser = argparse.ArgumentParser()
    stage_parser.add_argument('--stage', type=int, choices=[1, 2, 3],
                              help='Select specific training stage (1, 2, or 3). If not specified, runs all stages.')
    stage_args, _ = stage_parser.parse_known_args()

    from peft import LoraConfig
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    from training_utils import pretrain_tokenize_function, instruct_ft_tokenize_function
    from safetensors.torch import load_file
    from copy import deepcopy

    # 1) Stage 1
    def run_stage_1(prev_model=None, prev_dataset=None):
        return train_stage(
            model_args=model_args,
            training_args=training_args,
            pipeline_config=pipeline_config,
            tokenizer_func=pretrain_tokenize_function,
            data_path_train=pipeline_config.target_q_to_eng_q_path,
            data_path_eval=pipeline_config.eval_target_q_to_eng_q_path,
            num_epochs=pipeline_config.q_only_epochs,
            use_r_tokens=False,
            prev_model=prev_model,
            prev_dataset=prev_dataset,
            ewc=pipeline_config.use_continual_learning,
            ewc_lambda=pipeline_config.ewc_lambda,
            load_checkpoint=(pipeline_config.checkpoint_stage_1 if os.path.exists(pipeline_config.checkpoint_stage_1) else None),
            lora_config=lora_config,
            stage_name="Stage 1"
        )

    # 2) Stage 2
    def run_stage_2(prev_model=None, prev_dataset=None):
        return train_stage(
            model_args=model_args,
            training_args=training_args,
            pipeline_config=pipeline_config,
            tokenizer_func=pretrain_tokenize_function,
            data_path_train=pipeline_config.target_q_to_eng_r_path,
            data_path_eval=pipeline_config.eval_target_q_to_eng_r_path,
            num_epochs=pipeline_config.q_r_epochs,
            use_r_tokens=True,
            prev_model=prev_model,
            prev_dataset=prev_dataset,
            ewc=pipeline_config.use_continual_learning,
            ewc_lambda=pipeline_config.ewc_lambda,
            load_checkpoint=(pipeline_config.checkpoint_stage_2 if os.path.exists(pipeline_config.checkpoint_stage_2) else None),
            lora_config=lora_config,
            stage_name="Stage 2"
        )

    # 3) Stage 3
    def run_stage_3(prev_model=None, prev_dataset=None):
        return train_stage(
            model_args=model_args,
            training_args=training_args,
            pipeline_config=pipeline_config,
            tokenizer_func=instruct_ft_tokenize_function,
            data_path_train=pipeline_config.target_q_to_target_r_path,
            data_path_eval=pipeline_config.eval_target_q_to_target_r_path,
            num_epochs=pipeline_config.target_epochs,
            use_r_tokens=True,
            prev_model=prev_model,
            prev_dataset=prev_dataset,
            ewc=False,  # 보통 stage 3는 그냥 FT
            load_checkpoint=(pipeline_config.checkpoint_stage_3 if os.path.exists(pipeline_config.checkpoint_stage_3) else None),
            lora_config=lora_config,
            stage_name="Stage 3"
        )

    # 실행 로직
    if pipeline_config.run_stage_1 or pipeline_config.run_all_stages:
        model, train_dataset = run_stage_1()
        prev_model, prev_dataset = model, train_dataset
    else:
        prev_model, prev_dataset = None, None

    if pipeline_config.run_stage_2 or pipeline_config.run_all_stages:
        model, train_dataset = run_stage_2(prev_model, prev_dataset)
        prev_model, prev_dataset = model, train_dataset

    if pipeline_config.run_stage_3 or pipeline_config.run_all_stages:
        run_stage_3(prev_model, prev_dataset)


if __name__ == "__main__":
    train_pipeline()

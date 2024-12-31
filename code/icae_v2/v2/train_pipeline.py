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
    
    import argparse
    stage_parser = argparse.ArgumentParser()
    stage_parser.add_argument('--stage', type=int, choices=[1, 2, 3], help='Select specific training stage (1, 2, or 3). If not specified, runs all stages.')
    stage_args, _ = stage_parser.parse_known_args()
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    def train_stage_1():
        """Stage 1: Q토큰만 사용한 학습"""
        print("Stage 1: Training with Q tokens only")
        stage_training_args = deepcopy(training_args)
        stage_training_args.num_train_epochs = pipeline_config.q_only_epochs
        stage_training_args.use_r_tokens = False  # Stage 1에서는 R토큰 사용 안 함
        
        model = ICAE(model_args, stage_training_args, lora_config)
        
        # Train dataset
        memory_size = training_args.fixed_q_mem_size
        MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))
        train_dataset = load_dataset("json", data_files={"train": pipeline_config.target_q_to_eng_q_path})["train"]
        train_dataset = train_dataset.map(pretrain_tokenize_function, batched=True, batch_size=stage_training_args.per_device_train_batch_size, fn_kwargs={"model": model, "mem": MEM_TOKENS, "lm_ratio": training_args.lm_ratio})
        # Eval dataset
        eval_dataset = None
        if pipeline_config.eval_target_q_to_eng_q_path:
            eval_dataset = load_dataset("json", data_files={"validation": pipeline_config.eval_target_q_to_eng_q_path})["validation"]
            eval_dataset = eval_dataset.map(pretrain_tokenize_function, batched=True, fn_kwargs={"model": model, "mem": MEM_TOKENS})   # don't add lm in the dev set.
        data_collator = DataCollatorForDynamicPadding(model.pad_token_id)
        model = train_model(model, train_dataset, eval_dataset, stage_training_args, data_collator)
        return model, train_dataset

    def train_stage_2(prev_model=None, prev_dataset=None):
        """Stage 2: Q + R토큰 학습"""
        print("Stage 2: Training with Q + R tokens")
        stage_training_args = deepcopy(training_args)
        stage_training_args.num_train_epochs = pipeline_config.q_r_epochs
        stage_training_args.use_r_tokens = True  # Stage 2에서는 R토큰 사용
        
        model = ICAE(model_args, stage_training_args, lora_config)
        
        if prev_model is None and os.path.exists(pipeline_config.q_only_checkpoint):
            state_dict = load_file(os.path.join(pipeline_config.q_only_checkpoint, "model.safetensors"))
            model.load_state_dict(state_dict, strict=False)
        elif prev_model is not None:
            model.load_state_dict(prev_model.state_dict(), strict=False)

        memory_size = training_args.fixed_q_mem_size + training_args.fixed_r_mem_size
        MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))
        # Train dataset
        train_dataset = load_dataset("json", data_files={"train": pipeline_config.target_q_to_eng_r_path})["train"]
        train_dataset = train_dataset.map(pretrain_tokenize_function, batched=True, batch_size=stage_training_args.per_device_train_batch_size, fn_kwargs={"model": model, "mem": MEM_TOKENS, "lm_ratio": training_args.lm_ratio})
        
        # Eval dataset
        eval_dataset = None
        if pipeline_config.eval_target_q_to_eng_r_path:
            eval_dataset = load_dataset("json", data_files={"validation": pipeline_config.eval_target_q_to_eng_r_path})["validation"]
            eval_dataset = eval_dataset.map(pretrain_tokenize_function, batched=True, fn_kwargs={"model": model, "mem": MEM_TOKENS})   # don't add lm in the dev set.

        if pipeline_config.use_continual_learning and (prev_model is not None or os.path.exists(pipeline_config.q_only_checkpoint)):
            ewc = EWC(prev_model if prev_model is not None else model, prev_dataset if prev_dataset is not None else train_dataset)
            original_loss_fn = model.loss_fct
            def new_loss_fn(pred, target):
                return original_loss_fn(pred, target) + pipeline_config.ewc_lambda * ewc.penalty(model)
            model.loss_fct = new_loss_fn
        data_collator = DataCollatorForDynamicPadding(model.pad_token_id)
        model = train_model(model, train_dataset, eval_dataset, stage_training_args, data_collator)
        return model, train_dataset

    def train_stage_3(prev_model=None, prev_dataset=None):
        """Stage 3: 타겟 언어 파인튜닝"""
        print("Stage 3: Target language fine-tuning")
        stage_training_args = deepcopy(training_args)
        stage_training_args.num_train_epochs = pipeline_config.target_epochs
        
        model = ICAE(model_args, stage_training_args, lora_config)
        
        if prev_model is None and os.path.exists(pipeline_config.q_r_checkpoint):
            state_dict = load_file(os.path.join(pipeline_config.q_r_checkpoint, "model.safetensors"))
            model.load_state_dict(state_dict, strict=False)
        elif prev_model is not None:
            model.load_state_dict(prev_model.state_dict(), strict=False)
        memory_size = training_args.fixed_q_mem_size + training_args.fixed_r_mem_size
        MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))
        # Train dataset
        train_dataset = load_dataset("json", data_files={"train": pipeline_config.target_q_to_target_r_path})["train"]
        train_dataset = train_dataset.map(instruct_ft_tokenize_function, batched=True, batch_size=stage_training_args.per_device_train_batch_size, fn_kwargs={"model": model, "mem": MEM_TOKENS})
        # Eval dataset
        eval_dataset = None
        if pipeline_config.eval_target_q_to_target_r_path:
            eval_dataset = load_dataset("json", data_files={"validation": pipeline_config.eval_target_q_to_target_r_path})["validation"]
            eval_dataset = eval_dataset.map(instruct_ft_tokenize_function, batched=True, fn_kwargs={"model": model, "mem": MEM_TOKENS})
        data_collator = DataCollatorForDynamicPadding(model.pad_token_id)
        model = train_model(model, train_dataset, eval_dataset, stage_training_args, data_collator)

    if pipeline_config.run_stage_1 or pipeline_config.run_all_stages:
        model, train_dataset = train_stage_1()
        prev_model, prev_dataset = model, train_dataset
    else:
        prev_model, prev_dataset = None, None
    
    if pipeline_config.run_stage_2 or pipeline_config.run_all_stages:
        model, train_dataset = train_stage_2(prev_model, prev_dataset)
        prev_model, prev_dataset = model, train_dataset
    
    if pipeline_config.run_stage_3 or pipeline_config.run_all_stages:
        train_stage_3(prev_model, prev_dataset)

if __name__ == "__main__":
    train_pipeline()

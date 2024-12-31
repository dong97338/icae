from transformers import Trainer
import os
import torch
import random

from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import WandbCallback
import math


from safetensors.torch import save_file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_lora_and_token_embedding(model, output_dir):
    """
    LoRA 어댑터, memory_token_embed, AE/LM/FT 토큰 임베딩만 safetensors 형태로 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    filtered_state_dict = {}
    full_state_dict = model.state_dict()

    for k, v in full_state_dict.items():
        # (1) LoRA 어댑터
        #    일반적으로 "lora_" 혹은 "lora"라는 문자열이 파라미터 키에 포함됩니다.
        #    실제 키가 어떻게 찍히는지 확인 후 패턴에 맞춰 수정할 수도 있습니다.
        if "lora" in k.lower():
            filtered_state_dict[k] = v.cpu()
            continue

        # (2) memory_token_embed (메모리 토큰 임베딩)
        if "memory_token_embed" in k:
            filtered_state_dict[k] = v.cpu()
            continue

        # (3) AE/LM/FT 토큰이 포함된 임베딩 (즉, 늘어난 토큰 범위를 포함하는 임베딩)
        #     예시에서는 icae 모델 내부의 embed_tokens.weight 키가 "icae.get_base_model().model.embed_tokens.weight" 
        #     혹은 "icae.model.embed_tokens.weight" 정도로 나올 것입니다.
        #
        #     만약 전체 임베딩 레이어를 통으로 저장해도 괜찮다면, 아래 조건으로 걸러내 저장합니다.
        #     (추가로 Q/R 메모리 토큰 영역만 골라내어 저장하고 싶다면 slice 작업을 해도 되지만, 
        #      로딩 시 strict=False 로 로드하면 문제없으므로 전체를 저장해도 괜찮습니다.)
        if "embed_tokens.weight" in k.lower():
            filtered_state_dict[k] = v.cpu()
            continue

    # safetensors로 저장
    save_file(filtered_state_dict, os.path.join(output_dir, "model.safetensors"))
    print(f"LoRA + memory token + special tokens embedding만 저장 완료: {os.path.join(output_dir, 'model.safetensors')}")


def train_model(model, train_dataset, eval_dataset, training_args, data_collator=None):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # checkpoint을 설정
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    else:
        checkpoint = last_checkpoint

    if max(training_args.per_device_train_batch_size, training_args.per_device_eval_batch_size) == 1:
        data_collator = None

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(training_args)

    training_args.save_strategy = "no"
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = 1000

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    trainer.add_callback(WandbCallback())

    # 2) 학습
    if checkpoint is not None:
        print(f"Loaded from the checkpoint: {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    else:
        train_result = trainer.train()

    # 3) 평가
    trainer.log_metrics("train", train_result.metrics)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # 4) 필요한 파라미터(LoRA + memory token + special tokens embedding)만 safetensors로 저장
    save_lora_and_token_embedding(model, training_args.output_dir)

    return trainer.model



def text_extraction(input_ids, length, lm_ratio=0.0):
    
    input_len = len(input_ids)
    assert input_len >= 1, f"Error: invalid input length ({input_len})"
    
    # ae
    if random.random() >= lm_ratio: 
        if input_len <= length: # if shorter, keep the complete text
            return input_ids, []
        else:
            last_start = input_len - length
            random_start = random.randint(0, last_start)
            return input_ids[random_start: random_start+length], []
    
    # lm    
    if input_len <= length:
        r = random.randint(0, input_len-1)
        return input_ids[:r+1], input_ids[r+1:]
    else:
        last_start = input_len - length
        random_start = random.randint(0, last_start)
        return input_ids[random_start: random_start+length], input_ids[random_start+length:]


def pretrain_tokenize_function(examples, model, mem, lm_ratio=0.0):
    text_output = model.tokenizer(
        examples["input"], 
        truncation=True,
        padding="max_length",  # max_length=1024
        max_length=1024,
        return_attention_mask=False
    )
    label_output = model.tokenizer(
        examples["output"], 
        truncation=True,
        padding="max_length",  # max_length=512
        max_length=512,
        return_attention_mask=False
    )
    text_output['prompt_answer_ids'] = []
    text_output['labels'] = []

    max_len = model.training_args.model_max_length  # heuristic

    for idx in range(len(text_output["input_ids"])): # batch_size만큼 반복
        
        ae = True
        a, b = text_extraction(label_output["input_ids"][idx], max_len, lm_ratio=lm_ratio)
        length_a = len(a)
        num_segments = model.compute_num_segments(length_a)
        total_mem_length = num_segments * model.total_mem_size # 스테이지별로 달라져야함
        
        # if len(b) > model.training_args.min_tokens_for_lm:  # avoid too few tokens for lm, which is a waste of computing
        #     ae = False
        #     b = b[:max_len]

        # text_output['input_ids'][idx] = a

        # decoder part: note that in v2, we add mem_tokens to the prompt_ids for easy implementation; which is different from v1 implementation where mem tokens are not in the prompt_ids
        if ae:  # autoencoding objective
            prompt_ids = [mem[0]] * total_mem_length + [model.ae_token_id] # 스테이지별로 달라져야함
            answer_ids = a + [model.eos_id]    # if ae, eos token
        # else:   # lm objective
        #     prompt_ids = [mem[0]] * total_mem_length
        #     if model.training_args.add_special_token_for_lm:
        #         prompt_ids += [model.lm_token_id]
        #     answer_ids = b   # if lm, no eos token

        text_output['prompt_answer_ids'].append(prompt_ids + answer_ids)
        if ae:
            labels = [-100] * len(prompt_ids) + answer_ids
        # else:
        #     labels = [-100] * len(prompt_ids) + [-100] * model.training_args.leave_tokens_for_lm + answer_ids[model.training_args.leave_tokens_for_lm:] # no loss for leave_tokens_for_lm
        text_output['labels'].append(labels)
        assert len(text_output['prompt_answer_ids'][-1]) == len(labels)
        
    return text_output


def instruct_ft_tokenize_function(examples, model, mem):
    text_output = model.tokenizer(examples["input"], max_length=5120, truncation=True, padding=False, return_attention_mask=False, add_special_tokens=False)
    prompt_output = model.tokenizer(examples["prompt"], truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)
    label_output = model.tokenizer(examples["answer"], truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)
    text_output['prompt_answer_ids'] = []
    text_output['labels'] = []

    max_len = model.training_args.model_max_length  # heuristic

    for idx in range(len(text_output["input_ids"])):
        
        length = len(text_output["input_ids"][idx])
        num_segments = model.compute_num_segments(length)
        total_mem_length = num_segments * model.total_mem_size
        
        prompt_ids = [mem[0]] * total_mem_length + [model.ft_token_id] + prompt_output['input_ids'][idx]
        prompt_ids = [1, 733, 16289, 28793] + prompt_ids + [733, 28748, 16289, 28793]   # special formats for prompt in Mistral
        answer_ids = label_output['input_ids'][idx] + [model.eos_id]

        text_output['prompt_answer_ids'].append(prompt_ids + answer_ids)
            
        labels = [-100] * len(prompt_ids) + answer_ids
        text_output['labels'].append(labels)
        
        assert len(text_output['prompt_answer_ids'][-1]) == len(labels)
        
    return text_output


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
    def __call__(self, examples):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]
        prompt_answer_ids = [torch.tensor(example["prompt_answer_ids"], dtype=torch.long) for example in examples]
        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        prompt_answer_ids = self.dynamic_padding(prompt_answer_ids, fill_value=self.pad_token_id)
        labels = self.dynamic_padding(labels)
        batch = {"input_ids": input_ids, "labels": labels, "prompt_answer_ids": prompt_answer_ids}
        return batch
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences)
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences
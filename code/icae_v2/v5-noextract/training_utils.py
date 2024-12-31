# training_utils.py
from transformers import Trainer
import os
import torch
import random
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import WandbCallback
from safetensors.torch import save_file
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyTrainer(Trainer):
    """
    - evaluation 중에 이미 계산된 (logits, labels, prompt_answer_ids 등)을 저장해놓고
      on_evaluate 단계에서 그 중 하나만 디코딩해서 보여주는 예시 Trainer
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 매 evaluation마다 임시로 logits 등 누적할 리스트
        self.tmp_eval_storage = []

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None
    ):
        """
        Trainer가 evaluation(또는 predict) 시에 batch 단위로 호출하는 함수.
        여기서 추가로 (logits, prompt_answer_ids) 등을 저장해놓고,
        나중에 on_evaluate 시점에 디코딩에 활용.
        """
        # prediction_loss_only=True 면 기본적으로 logits를 반환하지 않는 것이 디폴트.
        # 하지만 아래처럼 compute_loss(..., return_outputs=True)를 써서 얻을 수 있게 함.
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            # outputs 안에 model.forward()가 리턴한 {"loss":..., "logits":..., ...} 등이 들어있다고 가정

        logits = outputs["logits"] if ("logits" in outputs) else None

        # logits가 있을 경우에만 저장
        if logits is not None and len(self.tmp_eval_storage) < 1:
            # inputs["prompt_answer_ids"] 등이 있을 수 있음 (pretrain_tokenize_function 참고)
            stored_item = {
                "logits": logits[0].detach().cpu(),  # (seq_len, vocab_size)
                "prompt_answer_ids": inputs["prompt_answer_ids"][0].detach().cpu() if "prompt_answer_ids" in inputs else None,
                "labels": inputs["labels"][0].detach().cpu() if "labels" in inputs else None,
            }
            self.tmp_eval_storage.append(stored_item)

        # Trainer 기본 반환값 구성
        # 예시로 (loss, None, None)
        if prediction_loss_only:
            return (loss, None, None)
        else:
            # logits가 필요하다면 (loss, logits, labels) 형태로 반환해야 함
            # 여기서는 간단히 (loss, None, None)만 반환
            return (loss, None, None)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Trainer가 evaluation(혹은 매 eval_steps마다) 끝났을 때 호출되는 함수.
        여기서 self.tmp_eval_storage 에 쌓인 logits 중 하나를 디코딩해서 보여줄 수 있음.
        """
        # 매번 evaluation 전, 임시 저장소 초기화
        self.tmp_eval_storage = []

        # ---------------------------
        # (중요) evaluation 시점만이라도 prediction_loss_only=False로 강제
        # ---------------------------
        backup_flag = self.args.prediction_loss_only
        self.args.prediction_loss_only = False

        # 원본 evaluate() 호출 → 내부적으로 prediction_loop가 돌면서 prediction_step()이 여러 번 불림
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        # 다시 되돌림
        self.args.prediction_loss_only = backup_flag

        # 이제 tmp_eval_storage 에 evaluation 동안의 샘플 1~N개가 모여 있을 수 있음
        if len(self.tmp_eval_storage) > 0:
            sample_item = self.tmp_eval_storage[0]
            logits = sample_item["logits"]  # (seq_len, vocab_size)
            prompt_answer_ids = sample_item["prompt_answer_ids"]  # (seq_len,)

            # teacher forcing 식 argmax
            pred_ids = logits.argmax(dim=-1)  # (seq_len,)
            text = self.model.tokenizer.decode(pred_ids, skip_special_tokens=True)

            print("=== [Evaluation Sample Decode] ===")
            print("Teacher forcing argmax decode:", text)
            print("===================================")

        return metrics


def save_lora_and_token_embedding(model, output_dir):
    """
    LoRA 어댑터, memory_token_embed, AE/LM/FT 토큰 임베딩만 safetensors 형태로 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    filtered_state_dict = {}
    full_state_dict = model.state_dict()

    for k, v in full_state_dict.items():
        # (1) LoRA 어댑터
        if "lora" in k.lower():
            filtered_state_dict[k] = v.cpu()
            continue

        # (2) memory_token_embed (메모리 토큰 임베딩)
        if "memory_token_embed" in k:
            filtered_state_dict[k] = v.cpu()
            continue

        # (3) 임베딩 레이어(토큰 임베딩) 전체
        if "embed_tokens.weight" in k.lower():
            filtered_state_dict[k] = v.cpu()
            continue

    # safetensors로 저장
    save_file(filtered_state_dict, os.path.join(output_dir, "model.safetensors"))
    print(f"LoRA + memory token + special tokens embedding만 저장 완료: {os.path.join(output_dir, 'model.safetensors')}")



def train_model(model, train_dataset, eval_dataset, training_args):
    """
    DataCollatorForDynamicPadding을 사용하지 않고,
    이미 tokenizer 단계에서 모든 길이가 동일하도록 padding="max_length" 처리를 했다고 가정합니다.
    """
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

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    else:
        checkpoint = last_checkpoint

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(training_args)

    # ----------------------------------------------------------------
    # Trainer 설정: 매 100 step마다 evaluation 수행
    # ----------------------------------------------------------------
    training_args.save_strategy = "no"
    training_args.evaluation_strategy = "steps"
    training_args.eval_steps = 100
    # 필요하다면 num_train_epochs, max_steps, logging_steps 등도 함께 조정

    # MyTrainer 사용
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=None  # 이미 padding="max_length" 가정
    )
    trainer.add_callback(WandbCallback())

    # 2) 학습
    if checkpoint is not None:
        print(f"Loaded from the checkpoint: {checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    else:
        train_result = trainer.train()

    # -----------------------
    # (학습 종료 후) 최종 평가
    # -----------------------
    # metrics만 계산
    final_metrics = trainer.evaluate()
    trainer.log_metrics("eval", final_metrics)
    trainer.save_metrics("eval", final_metrics)

    # (필요시) 추가로 predict() 이용하여 logits / label_ids 확인 가능
    # pred_results = trainer.predict(eval_dataset)
    # ...

    # 마지막으로 필요한 파라미터만 safetensors로 저장
    save_lora_and_token_embedding(model, training_args.output_dir)
    return trainer.model

def pretrain_tokenize_function(examples, model, mem):
    """
    - 'input'과 'output'을 각각 토큰화하고,
    - 라벨 부분에서만 pad_token → -100 치환
    - 실제 모델 입력에는 pad_token_id를 그대로 둠
    """
    # (1) 인풋 토큰화
    text_output = model.tokenizer(
        examples["input"],
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_attention_mask=False
    )

    # (2) 라벨(=output) 토큰화
    label_output = model.tokenizer(
        examples["output"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_attention_mask=False
    )

    text_output["prompt_answer_ids"] = []
    text_output["labels"] = []

    pad_token_id = model.tokenizer.pad_token_id

    for idx in range(len(text_output["input_ids"])):
        # ---------------------
        # 1) 라벨 (output) 쪽 처리
        # ---------------------
        raw_label_ids = label_output["input_ids"][idx]
        # pad_token → -100으로 치환
        label_ids = [
            (-100 if token_id == pad_token_id else token_id)
            for token_id in raw_label_ids
        ]
        # 뒤에 EOS 추가
        label_ids_with_eos = label_ids + [model.eos_id]

        # ---------------------
        # 2) 세그먼트(메모리) 계산
        # ---------------------
        length_a = len(label_ids_with_eos)
        num_segments = model.compute_num_segments(length_a)
        total_mem_length = num_segments * model.total_mem_size

        # ---------------------
        # 3) 모델 "입력"에 들어갈 prompt_answer_ids
        #    → 이때 label_ids와 달리 pad_token_id 유지!
        # ---------------------
        prompt_ids = [mem[0]] * total_mem_length + [model.ae_token_id]
        # label_ids_with_eos 안의 -100은 임베딩 불가이므로,
        # 모델 입력 쪽은 여전히 pad_token_id를 사용해야 합니다.
        # => 아래처럼 label_ids_with_eos에서 -100을 다시 pad_token_id로 되돌리거나,
        #    애초부터 분리해서 구성해야 합니다.
        answer_input_ids = [
            (pad_token_id if t == -100 else t)
            for t in label_ids_with_eos
        ]

        # 최종 prompt+answer
        pa_input_ids = prompt_ids + answer_input_ids

        # ---------------------
        # 4) "라벨" 배열
        # ---------------------
        labels = ([-100] * len(prompt_ids)) + label_ids_with_eos

        # ---------------------
        # 5) 저장
        # ---------------------
        text_output["prompt_answer_ids"].append(pa_input_ids)
        text_output["labels"].append(labels)

    return text_output


def instruct_ft_tokenize_function(examples, model, mem):
    text_output = model.tokenizer(examples["input"], padding="max_length", max_length=512, truncation=True, return_attention_mask=False)
    prompt_tokenized = model.tokenizer(examples["prompt"], padding=False, truncation=False, return_attention_mask=False)
    answer_tokenized = model.tokenizer(examples["answer"], padding=False, truncation=False, return_attention_mask=False)
    text_output["prompt_answer_ids"], text_output["labels"] = [], []
    maximum_length = 1536
    for index in range(len(text_output["input_ids"])):
        length_of_input_ids = len(text_output["input_ids"][index])
        number_of_segments = model.compute_num_segments(length_of_input_ids)
        total_memory_length = number_of_segments * model.total_mem_size
        prompt_ids = [mem[0]] * total_memory_length + [model.ft_token_id] + prompt_tokenized["input_ids"][index]
        prompt_ids = [1, 733, 16289, 28793] + prompt_ids + [733, 28748, 16289, 28793]
        answer_ids = answer_tokenized["input_ids"][index] + [model.eos_id]
        combined_ids = prompt_ids + answer_ids
        if len(combined_ids) > maximum_length: combined_ids = combined_ids[:maximum_length]
        labels = [-100] * len(prompt_ids) + answer_ids
        if len(labels) > maximum_length: labels = labels[:maximum_length]
        if len(combined_ids) < maximum_length:
            padding_length = maximum_length - len(combined_ids)
            combined_ids += [model.pad_token_id] * padding_length
            labels += [-100] * padding_length
        text_output["prompt_answer_ids"].append(combined_ids)
        text_output["labels"].append(labels)
    return text_output

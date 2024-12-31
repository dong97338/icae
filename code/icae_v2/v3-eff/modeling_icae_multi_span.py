# ICAE that supports multi span concat

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import torch
import torch.nn as nn
import random
from dataclasses import dataclass, field
from typing import Optional
from peft import (
    get_peft_model,
)
from torch.nn.functional import gelu
import math
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    train: bool = field(
        default=True,
        metadata={"help": "if true, the model ckpt will be initialized for training; else, it's for inference"}
    )
    lora_r: int = field(
        default=128,
        metadata={"help": "LoRA rank"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    debug_data: bool = field(default=False, metadata={"help": "Enable debug dataset to quickly verify the training process"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    fixed_q_mem_size: int = field(
        default=128,
        metadata={"help": "Fixed memory size for Q tokens during training"}
    )
    fixed_r_mem_size: int = field(
        default=128,
        metadata={"help": "Fixed memory size for R tokens during training"}
    )
    use_r_tokens: bool = field(
        default=False,
        metadata={"help": "Whether to use R tokens in addition to Q tokens"}
    )
    mean_compression_rate: int = field(
        default=4,
        metadata={"help": "Mean compression rate; default=4"},
    )
    min_tokens_for_lm: int = field(
        default=64,
        metadata={"help": "Minimum tokens for lm objective learning"},
    )
    leave_tokens_for_lm: int = field(
        default=8,
        metadata={"help": "Leave some tokens without loss for lm objective"},
    )
    lm_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio for LM training."},
    )
    add_special_token_for_lm: bool = field(
        default=False,
        metadata={"help": "Add a special token for the prompt of language modeling; default: False"},
    )
    restore_from: str = field(
        default="",
        metadata={"help": "The checkpoint that should be restored from for fine-tuning"}
    )

def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")

def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class ICAE(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        
        # bfloat16으로 고정
        self.icae = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            resume_download=True
        )
        
        self.training = self.model_args.train    
        
        if self.training:
            self.decoder = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="auto",
                resume_download=True
            )

        self.vocab_size = self.icae.config.vocab_size + 1    # [PAD] token
        self.pad_token_id = self.vocab_size - 1
        self.mean_compression_rate = training_args.mean_compression_rate

        # 메모리 크기 설정
        self.q_mem_size = training_args.fixed_q_mem_size
        self.r_mem_size = training_args.fixed_r_mem_size
        
        # 토큰 임베딩 초기화 - 하나의 통합된 임베딩 레이어 사용
        self.memory_token_embed = nn.Embedding(
            self.q_mem_size + self.r_mem_size + 3,
            self.icae.config.hidden_size,
            padding_idx=None
        )
        
        self.vocab_size_with_mem = self.vocab_size + self.q_mem_size + self.r_mem_size
        # special tokens in addition to mem and length tokens
        self.ae_token_id = self.vocab_size_with_mem + 0
        self.lm_token_id = self.vocab_size_with_mem + 1
        self.ft_token_id = self.vocab_size_with_mem + 2

        self.icae.resize_token_embeddings(self.vocab_size_with_mem + 3) 
        
        # special tokens for Llama-2/Mistral tokenizer
        self.bos_id = 1
        self.eos_id = 2
        
        self.dim = self.icae.config.hidden_size
        self.icae = get_peft_model(self.icae, lora_config)
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        
        # Q토큰과 R토큰을 위한 sequence 생성
        if self.training_args.use_r_tokens:
            self.append_sequence = torch.arange(
                self.vocab_size,
                self.vocab_size + self.q_mem_size + self.r_mem_size,
                dtype=torch.long, device=device
            ).unsqueeze(0)
        else:
            self.append_sequence = torch.arange(
                self.vocab_size,
                self.vocab_size + self.q_mem_size,
                dtype=torch.long, device=device
            ).unsqueeze(0)
        
        if self.training:
            self.init()

    @property
    def total_mem_size(self):
        return self.q_mem_size + (self.r_mem_size if self.training_args.use_r_tokens else 0)

    def init(self):
        print("Freezing the decoder...")
        freeze_model(self.decoder)
        self.decoder.eval()
        print_trainable_parameters(self)
        
        if self.training_args.restore_from is not None and self.training_args.restore_from != "":
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")
        
        print("Enabling gradient checkpointing...")
        # self.icae.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                
    def compute_num_segments(self, total_length):
        divisor = self.total_mem_size
        num_segments = math.ceil(total_length / divisor / self.mean_compression_rate)
        return max(1, num_segments)  # 최소 1개의 세그먼트 보장

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        prompt_answer_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # Encoder part
        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)
        
        prompt_answer_embs = self.icae.get_base_model().model.embed_tokens(prompt_answer_ids)
        max_compressed_length = num_segments * self.total_mem_size
        compress_outputs = torch.zeros(
            (batch_size, max_compressed_length, self.dim),
            dtype=prompt_answer_embs.dtype,
            device=prompt_answer_embs.device
        )

        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            
            # Expand self.append_sequence to match batch size
            append_sequence = self.append_sequence.expand(batch_size, -1)
            segment_input_ids = torch.cat([segment_input_ids, append_sequence], dim=1)
            
            mem_flag = segment_input_ids >= self.vocab_size
            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            segment_input_embedding[mem_flag] = self.memory_token_embed(
                segment_input_ids[mem_flag] - self.vocab_size
            ).to(segment_input_embedding)
            
            segment_compress_outputs = self.icae(
                inputs_embeds=segment_input_embedding,
                output_hidden_states=True
            )
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]
            
            for i in range(batch_size):
                mem_positions = mem_flag[i].nonzero(as_tuple=False).squeeze(1)
                compress_outputs[i, segment_idx * self.total_mem_size : (segment_idx + 1) * self.total_mem_size, :] = segment_compress_outputs[i, mem_positions, :]
            
            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()
        
        # Decoder part
        decoder_mem_flag = (prompt_answer_ids >= self.vocab_size) & (prompt_answer_ids < self.vocab_size + self.total_mem_size)
        
        for i in range(batch_size):
            num_mem_tokens = decoder_mem_flag[i].sum().item()
            assert max_compressed_length >= num_mem_tokens, f"Not enough mem tokens to assign for batch {i}"
            prompt_answer_embs[i, decoder_mem_flag[i]] = compress_outputs[i, :num_mem_tokens, :]
        
        special_prompt = prompt_answer_ids >= self.vocab_size_with_mem
        prompt_answer_embs[special_prompt] = self.memory_token_embed(
            prompt_answer_ids[special_prompt] - self.vocab_size
        ).to(prompt_answer_embs)
        
        if self.training:
            decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
        else:
            with self.icae.disable_adapter():
                decoder_outputs = self.icae(inputs_embeds=prompt_answer_embs, output_hidden_states=True)

        logits = decoder_outputs.logits
        effective_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        target_ids = labels[:, 1:].reshape(-1)
        loss = self.loss_fct(effective_logits, target_ids)
        return {"loss": loss, "logits": logits}

    def tokens_to_embeddings(self, token_ids):
        embeddings = self.icae.get_base_model().model.embed_tokens(token_ids)
        special_flags = token_ids >= self.vocab_size
        embeddings[special_flags] = self.memory_token_embed(
            token_ids[special_flags] - self.vocab_size
        ).to(embeddings)
        return embeddings
        
    def _compress(self, input_ids: torch.LongTensor):
        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)

        max_compressed_length = num_segments * self.total_mem_size
        compress_outputs = torch.zeros((batch_size, max_compressed_length, self.dim), device=input_ids.device)

        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            append_sequence = self.append_sequence.expand(batch_size, -1)
            segment_input_ids = torch.cat([segment_input_ids, append_sequence], dim=1)
            mem_flag = segment_input_ids >= self.vocab_size

            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            segment_input_embedding[mem_flag] = self.memory_token_embed(
                segment_input_ids[mem_flag] - self.vocab_size
            ).to(segment_input_embedding)

            segment_compress_outputs = self.icae(
                inputs_embeds=segment_input_embedding,
                output_hidden_states=True
            )
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]

            for i in range(batch_size):
                mem_positions = mem_flag[i].nonzero(as_tuple=False).squeeze(1)
                if mem_positions.numel() != self.total_mem_size:
                    raise ValueError(f"Expected {self.total_mem_size} mem positions, but got {mem_positions.numel()} for batch {i}")
                compress_outputs[i, segment_idx * self.total_mem_size : (segment_idx + 1) * self.total_mem_size, :] = segment_compress_outputs[i, mem_positions, :]

            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()

        return compress_outputs

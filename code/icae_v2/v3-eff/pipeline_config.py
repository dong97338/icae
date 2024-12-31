from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    
    # 실행할 스테이지 설정
    run_all_stages: bool = field(
        default=True,
        metadata={"help": "Run all training stages"}
    )
    run_stage_1: bool = field(
        default=False,
        metadata={"help": "Run stage 1 (Q-token only training)"}
    )
    run_stage_2: bool = field(
        default=False,
        metadata={"help": "Run stage 2 (Q+R token training)"}
    )
    run_stage_3: bool = field(
        default=False,
        metadata={"help": "Run stage 3 (target language fine-tuning)"}
    )
    
    # 스테이지별 에포크 수
    q_only_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs for Q-token only training"}
    )
    q_r_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs for Q+R token training"}
    )
    target_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs for target language fine-tuning"}
    )
    
    # 체크포인트 경로
    q_r_checkpoint: str = field(default="/ssd1/donghyeon/icae/out/1210-mintest")
    # target_checkpoint_path: str = field(init=False)
    
    data_folder: str = field(
        default="data",
        metadata={"help": "데이터셋 파일들이 위치한 기본 폴더"}
    )
    
    # 훈련 데이터셋 경로들
    target_q_to_eng_q_path: str = field(init=False)
    target_q_to_eng_r_path: str = field(init=False)
    target_q_to_target_r_path: str = field(init=False)
    
    # 평가 데이터셋 경로들
    eval_target_q_to_eng_q_path: Optional[str] = field(init=False)
    eval_target_q_to_eng_r_path: Optional[str] = field(init=False)
    eval_target_q_to_target_r_path: Optional[str] = field(init=False)
    
    # 평가 데이터셋 경로
    eval_target_q_to_eng_q_path: Optional[str] = field(
        default="data/stage1_eval.jsonl",
        metadata={"help": "Path to target_q to eng_q evaluation data"}
    )
    eval_target_q_to_eng_r_path: Optional[str] = field(
        default="data/stage2_eval.jsonl",
        metadata={"help": "Path to target_q to eng_r evaluation data"}
    )
    eval_target_q_to_target_r_path: Optional[str] = field(
        default="data/stage3_eval.jsonl",
        metadata={"help": "Path to target_q to target_r evaluation data"}
    )
    
    # 연속 학습 설정
    use_continual_learning: bool = field(
        default=False,
        metadata={"help": "Use continual learning with EWC"}
    )
    ewc_lambda: float = field(
        default=0.5,
        metadata={"help": "EWC regularization strength"}
    )

    def __post_init__(self):
        # 훈련 데이터셋 경로 설정
        self.target_q_to_eng_q_path = os.path.join(self.data_folder, "stage1_train.jsonl")
        self.target_q_to_eng_r_path = os.path.join(self.data_folder, "stage2_train.jsonl")
        self.target_q_to_target_r_path = os.path.join(self.data_folder, "stage3_train.jsonl")
        
        # 평가 데이터셋 경로 설정
        self.eval_target_q_to_eng_q_path = os.path.join(self.data_folder, "stage1_eval.jsonl")
        self.eval_target_q_to_eng_r_path = os.path.join(self.data_folder, "stage2_eval.jsonl")
        self.eval_target_q_to_target_r_path = os.path.join(self.data_folder, "stage3_eval.jsonl")

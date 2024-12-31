# pipeline_config.py
from dataclasses import dataclass, field
from typing import Optional, List
import os

@dataclass
class PipelineConfig:
    """
    3개 스테이지에 대해
      - run_stages[i]: 스테이지 i 실행 여부
      - epochs[i]: 스테이지 i 학습 에폭 수
      - train_files[i], eval_files[i]: 스테이지 i의 학습/검증 데이터 경로
    등을 리스트로 관리.
    """

    # 새로운 필드: 데이터가 들어있는 상위 디렉토리
    data: str = field(
        default="mini_data",
        metadata={"help": "Folder that contains stage{i}_train.jsonl, stage{i}_eval.jsonl, etc."}
    )

    # 각 스테이지별 epoch 수
    epochs: List[int] = field(
        default_factory=lambda: [1, 1, 1],
        metadata={"help": "Number of epochs for each stage (stage1~3). If 0, that stage is skipped."}
    )

    # 스테이지별 학습/평가 데이터 경로 (미리 지정해 두되, __post_init__에서 동적으로 채움)
    train_files: List[str] = field(
        default_factory=list,
        metadata={"help": "List of train dataset paths for each stage (stage1~3)"}
    )
    eval_files: List[str] = field(
        default_factory=list,
        metadata={"help": "List of eval dataset paths for each stage (stage1~3)"}
    )

    # (기존) 연속 학습(EWC) 설정 등 필요시 유지
    use_continual_learning: bool = field(
        default=False,
        metadata={"help": "Use continual learning with EWC"}
    )
    ewc_lambda: float = field(
        default=0.5,
        metadata={"help": "EWC regularization strength"}
    )

    def __post_init__(self):
        """
        사용자가 train_files / eval_files 를 명시적으로 주지 않았다면,
        self.data 경로를 기준으로 stage1_train.jsonl, stage2_train.jsonl, etc. 를 설정한다.
        """
        if not self.train_files:
            # 예: data=mini_data -> mini_data/stage1_train.jsonl, mini_data/stage2_train.jsonl, mini_data/stage3_train.jsonl
            self.train_files = [os.path.join(self.data, f"stage{i}_train.jsonl") for i in range(1, 4)]
        if not self.eval_files:
            # 예: data=mini_data -> mini_data/stage1_eval.jsonl, mini_data/stage2_eval.jsonl, mini_data/stage3_eval.jsonl
            self.eval_files = [os.path.join(self.data, f"stage{i}_eval.jsonl") for i in range(1, 4)]


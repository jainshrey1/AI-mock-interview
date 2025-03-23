from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    model_name: str
    tokenizer_name: str

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    tokenizer_name: Path
    output_dir: Path
    num_train_epochs: int
    max_steps: int
    learning_rate: float
    optim: str
    warmup_steps: int
    per_device_train_batch_size: int
    weight_decay: float
    logging_steps: int
    logging_dir: Path
    save_strategy: str
    save_steps: int
    evaluation_strategy: str
    eval_steps: int
    do_eval: bool
    report_to: None
    overwrite_output_dir: bool
    group_by_length: bool
    gradient_checkpointing: bool
    gradient_accumulation_steps: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
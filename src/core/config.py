from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str = "huihui-ai/Phi-4-mini-instruct-abliterated"

@dataclass
class TrainingConfig:
    batch_size: int = 2
    epochs: int = 3
    lr: float = 2e-4
    max_length: int = 512

@dataclass
class ProjectConfig:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
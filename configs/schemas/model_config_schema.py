from pydantic import BaseModel, Field
from typing import List, Dict, Union

class BaselineModelParameters(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    random_state: int

class FeatureEngineeringConfig(BaseModel):
    statistical_features: bool
    stability_features: bool
    duration_features: bool
    interaction_features: bool

class ValidationConfig(BaseModel):
    cv_folds: int
    early_stopping_rounds: int
    eval_metric: str

class BaselineModelConfig(BaseModel):
    algorithm: str
    parameters: BaselineModelParameters
    feature_engineering: FeatureEngineeringConfig
    validation: ValidationConfig

class LSTMArchitectureConfig(BaseModel):
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool

class LSTMTrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float
    num_epochs: int
    early_stopping_patience: int
    weight_decay: float

class DataProcessingConfig(BaseModel):
    sequence_length: int
    normalization: str
    padding: str

class LossFunctionConfig(BaseModel):
    type: str
    pos_weight: float

class LSTMModelConfig(BaseModel):
    architecture: LSTMArchitectureConfig
    training: LSTMTrainingConfig
    data_processing: DataProcessingConfig
    loss_function: LossFunctionConfig

class EvaluationConfig(BaseModel):
    metrics: List[str]
    test_size: float = Field(..., ge=0, le=1)
    stratify: bool
    random_state: int

class ModelConfig(BaseModel):
    baseline_model: BaselineModelConfig
    lstm_model: LSTMModelConfig
    evaluation: EvaluationConfig

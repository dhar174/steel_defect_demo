from pydantic import BaseModel, Field
from typing import List

class RealTimeSimulationConfig(BaseModel):
    playback_speed_multiplier: int
    update_interval_seconds: int
    buffer_size_seconds: int

class ThresholdsConfig(BaseModel):
    defect_probability: float = Field(..., ge=0, le=1)
    high_risk_threshold: float = Field(..., ge=0, le=1)
    alert_threshold: float = Field(..., ge=0, le=1)

class OutputConfig(BaseModel):
    log_predictions: bool
    save_trajectories: bool
    dashboard_enabled: bool
    dashboard_port: int

class InferenceConfig(BaseModel):
    model_types: List[str]
    real_time_simulation: RealTimeSimulationConfig
    thresholds: ThresholdsConfig
    output: OutputConfig

class MonitoringConfig(BaseModel):
    metrics_logging: bool
    performance_tracking: bool
    data_drift_detection: bool

class FullInferenceConfig(BaseModel):
    inference: InferenceConfig
    monitoring: MonitoringConfig

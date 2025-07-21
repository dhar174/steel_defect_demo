from pydantic import BaseModel, Field
from typing import Dict, List, Union

class SensorConfig(BaseModel):
    base_value: float
    noise_std: float
    min_value: float
    max_value: float

class DefectTriggers(BaseModel):
    prolonged_mold_level_deviation: int
    rapid_temperature_drop: int
    high_speed_with_low_superheat: bool

class DefectSimulationConfig(BaseModel):
    defect_probability: float = Field(..., ge=0, le=1)
    defect_triggers: DefectTriggers

class OutputConfig(BaseModel):
    raw_data_format: str
    metadata_format: str
    train_test_split: float = Field(..., ge=0, le=1)

class DataGenerationConfig(BaseModel):
    num_casts: int
    cast_duration_minutes: int
    sampling_rate_hz: int
    random_seed: int
    sensors: Dict[str, SensorConfig]
    defect_simulation: DefectSimulationConfig
    output: OutputConfig

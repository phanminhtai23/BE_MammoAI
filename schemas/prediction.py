from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import IntEnum


class PredictionRecord(BaseModel):
    """Schema chính để lưu dữ liệu dự đoán"""

    id: str  # UUID
    doctor_id: str
    prediction_date: datetime
    image_url: str
    image_key: str
    image_original_name: str
    prediction_result: str
    probability: float
    model_name: str

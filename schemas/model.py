from dataclasses import Field
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import IntEnum


class ModelInfor(BaseModel):
    """Schema chính để lưu thông tin model"""

    id: str  # UUID
    name: str  # Tên model
    version: Optional[str] = None  # Version model
    accuracy: Optional[float] = None  # Độ chính xác của model
    model_url: str  # URL S3 của model
    model_url: str
    model_key: str
    model_original_name: str
    is_active: bool = False  # Model có đang active không
    created_at: datetime
    updated_at: datetime


class ModelCreate(BaseModel):
    """Schema để tạo model mới"""

    name: str
    version: Optional[str] = None
    accuracy: Optional[float] = None  # Độ chính xác của model

    model_url: str
    model_key: str
    model_original_name: str
    is_active: bool = False


class ModelUpdate(BaseModel):
    """Schema để cập nhật model"""

    name: Optional[str] = None
    version: Optional[str] = None
    accuracy: Optional[float] = None  # Độ chính xác của model

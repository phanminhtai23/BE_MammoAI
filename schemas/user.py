from pydantic import BaseModel, EmailStr
from typing import Optional, Literal
from datetime import datetime


class UserSchema(BaseModel):
    id: str
    email: Optional[EmailStr]
    name: str
    auth_provider: Literal["local", "google", "facebook"] = "local"
    provider_id: Optional[str] = None
    isRevoked: bool = False
    confirmed: bool = False
    role: str = "user"
    created_at: datetime

class UserRegister(BaseModel):
    name: str  # Họ tên
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    device_info: Optional[str] = None


class UserResponse(BaseModel):
    id: str  # UUID
    email: EmailStr
    name: str
    auth_provider: Optional[str] = "local"
    provider_id: Optional[str] = None
    imgUrl: Optional[str] = None
    isRevoked: bool = False
    confirmed: bool = True
    role: str = "user"
    created_at: datetime


class UserInDB(BaseModel):
    """Schema cho user trong database"""
    id: str
    email: EmailStr
    password_hash: str  # Password đã được hash
    name: str
    auth_provider: Optional[str] = "local"
    provider_id: Optional[str] = None
    isRevoked: bool = False
    confirmed: bool = True
    role: str = "user"
    created_at: datetime


class UserUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    isRevoked: Optional[bool] = None
    confirmed: Optional[bool] = None


class UserPublic(BaseModel):
    """Schema cho response công khai (không có password)"""
    id: str
    email: EmailStr
    name: str
    role: str
    confirmed: bool
    created_at: datetime


class SendVerificationCode(BaseModel):
    email: EmailStr

class VerifyCode(BaseModel):
    email: EmailStr
    code: str


class VerifyResetToken(BaseModel):
    token: str

class ResetPassword(BaseModel):
    token: str
    new_password: str
    confirm_password: str
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class VerificationCode(BaseModel):
    email: EmailStr
    code: str
    created_at: datetime
    expires_at: datetime
    is_used: bool = False
    purpose: str = "email_verification"  # email_verification, password_reset, etc.

class VerifyCodeRequest(BaseModel):
    email: EmailStr
    code: str

class VerifyCodeResponse(BaseModel):
    success: bool
    message: str
    expires_at: Optional[datetime] = None
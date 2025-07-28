from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime


class UserSession(BaseModel):
    session_id: str
    user_id: str
    login_at: datetime
    logout_at: datetime
    is_active: bool

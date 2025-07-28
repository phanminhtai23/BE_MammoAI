from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from config import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)

# from database import tokens_collection
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer

security = HTTPBearer()


# Tạo JWT
def create_access_token(data: dict):
    to_encode = data.copy()
    created_at = datetime.now(timezone.utc)
    expires_at = created_at + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expires_at})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, created_at, expires_at


# Xác minh JWT
async def verify_access_token(token: str):
    try:
        # Kiểm tra token có hợp lệ không
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # Kiểm tra hạn token
        if payload["exp"] < datetime.now(timezone.utc).timestamp():
            return None

        return payload  # Token hợp lệ, trả về payload chứa thông tin user
    except JWTError:
        return None  # Token không hợp lệ


# Dependency để verify token và check admin role
async def verify_admin_token(token: str = Depends(security)):
    """Verify JWT token và check admin role"""
    try:
        # Decode JWT token
        payload = await verify_access_token(token.credentials)
        if not payload:
            raise HTTPException(
                status_code=401, detail="Token không hợp lệ hoặc đã hết hạn"
            )

        # Check admin role
        user_role = payload.get("role")
        if user_role != "admin":
            raise HTTPException(
                status_code=403, detail="Chỉ admin mới có quyền quản lý model"
            )

        return payload

    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Lỗi xác thực: {str(e)}")


async def verify_token(token: str = Depends(security)):
    """Verify JWT token và check admin role"""
    try:
        # print("voday")
        # Decode JWT token
        payload = await verify_access_token(token.credentials)
        if not payload:
            raise HTTPException(
                status_code=401, detail="Token không hợp lệ hoặc đã hết hạn"
            )

        return payload

    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Lỗi xác thực: {str(e)}")


# def create_refresh_token(user_id: str, device_info: str):
#     expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
#     refresh_token = jwt.encode(
#         {"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

#     # Lưu vào MongoDB
#     tokens_collection.insert_one({
#         "user_id": ObjectId(user_id),
#         "token": refresh_token,
#         "expires_at": expire,
#         "created_at": datetime.now(timezone.utc),
#         "device_info": device_info,
#         "is_revoked": False
#     })

#     return refresh_token

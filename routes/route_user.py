from fastapi import APIRouter, HTTPException, Depends
from database import users_collection, users_session_collection
from schemas.user import UserRegister, LoginRequest, UserUpdate, UserResponse, SendVerificationCode, VerifyCode, VerifyResetToken, ResetPassword
from utils.security import hash_password, verify_password, get_current_user, check_admin_role
from utils.jwt import create_access_token, verify_access_token
from fastapi.encoders import jsonable_encoder
import uuid
from datetime import datetime, timezone, timedelta  # Thêm timedelta vào import
from database import verification_codes_collection
# from utils.send_mail import send_verification_email, generate_verification_code, send_forgot_password_email
from services.send_mail import mail_service, generate_verification_code
from config import VERIFICATION_CODE_EXPIRE_MINUTES, FRONTEND_URL, GOOGLE_AUTH_URL

router = APIRouter()

# Helper function để tạo user session
async def create_user_session(user_id: str, auth_provider: str = "local", session_hours: int = 1):
    """
    Tạo session mới cho user khi đăng nhập
    
    Args:
        user_id: ID của user
        auth_provider: Loại đăng nhập (local/google/facebook)
        session_hours: Số giờ session có hiệu lực (mặc định 1 giờ)
    """
    session_id = str(uuid.uuid4())
    vietnam_tz = timezone(timedelta(hours=7))
    now = datetime.now(vietnam_tz)
    expires_at = now + timedelta(hours=session_hours)
    
    print("expires_at: ", expires_at)
    session_data = {
        "session_id": session_id,
        "user_id": user_id,
        "login_at": now,
        "expires_at": expires_at,  # Thời gian hết hạn session
        "logout_at": None,
        "is_active": True,
        "auth_provider": auth_provider
    }
    
    try:
        await users_session_collection.insert_one(session_data)
        print(f"✅ Tạo session: {session_id} (hết hạn: {expires_at})")
        return session_id
    except Exception as e:
        print(f"❌ Lỗi khi tạo session: {str(e)}")
        return None

# Helper function để cập nhật session khi logout
async def end_user_session(session_id: str):
    """
    Kết thúc session khi user logout
    """
    try:
        await users_session_collection.update_one(
            {"session_id": session_id, "is_active": True},
            {
                "$set": {
                    "logout_at": datetime.now(timezone.utc),
                    "is_active": False
                }
            }
        )
        return True
    except Exception as e:
        print(f"Lỗi khi cập nhật session: {str(e)}")
        return False

@router.post("/register", response_model=dict)
async def register(user: UserRegister):
    """
    Đăng ký tài khoản mới
    - **name**: Họ và tên
    - **email**: Email (phải unique)
    - **password**: Mật khẩu (sẽ được hash)
    """
    # Kiểm tra email đã tồn tại chưa
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=400, 
            detail="Email đã được đăng ký"
        )
    
    # Tạo UUID cho user
    user_id = str(uuid.uuid4())
    
    # Tạo dữ liệu user theo schema mới
    user_data = {
        "id": user_id,
        "email": user.email,
        "password_hash": hash_password(user.password),  # Hash password
        "name": user.name,
        "auth_provider": "local",  # Mặc định là local
        "provider_id": None,
        "isRevoked": False,
        "confirmed": False,  # Mặc định đã xác nhận
        "role": "user",  # Mặc định là user
        "created_at": datetime.now(timezone.utc)
    }
    
    try:
        # Lưu vào database
        result = await users_collection.insert_one(user_data)
        # Trả về thông tin user đã tạo
        return {
            "status_code": 201,
            "message": "Đăng ký tài khoản thành công",
            "data": {
                "user_id": user_id,
                "email": user.email,
                "name": user.name,
                "role": "user",
                "created_at": user_data["created_at"]
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tạo tài khoản: {str(e)}"
        )

@router.post("/send-verification-code")
async def send_verification_code(data: SendVerificationCode):
    """
    Gửi mã xác thực đến email
    """

    # Xóa mã xác thực cũ
    await verification_codes_collection.delete_many({"email": data.email})

    # Kiểm tra .email có tồn tại không
    existing_user = await users_collection.find_one({"email": data.email})
    if not existing_user:
        raise HTTPException(
            status_code=404,
            detail="Email không tồn tại"
        )
    
    # Tạo mã xác thực
    verification_code = generate_verification_code()


    # Lưu mã xác thực
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=VERIFICATION_CODE_EXPIRE_MINUTES)
    print("expires_at luu vao databaste:", expires_at)
    await verification_codes_collection.insert_one({
        "email": data.email,
        "code": verification_code,
        "expires_at": expires_at,
    })


    #Gửi mail
    expires_at_vn = expires_at.astimezone(timezone(timedelta(hours=7)))
    mail_service.send_verification_email(data.email, verification_code, expires_at_vn)

    return {"status": 200, "message": "Mã xác thực đã được gửi đến email"}


#verify mã xác nhận đăng ký tài khoản
@router.post("/verify-code")
async def verify_code(data: VerifyCode):
    """
    Xác thực mã xác thực
    """
    # print("data:", data)
    # Kiểm tra mã xác thực có tồn tại không
    verification_code = await verification_codes_collection.find_one({"email": data.email})

    if not verification_code:
        raise HTTPException(
            status_code=404,
            detail="Email chưa đăng ký hoặc đã xác thực tài khoản"
        )
    
    expires_at = verification_code["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
        
    now = datetime.now(timezone.utc)
    print("now:", now)
    print("expires_at:", expires_at)
    # Kiểm tra mã xác thực có hết hạn không
    if expires_at < now:
        raise HTTPException(
            status_code=400,
            detail="Mã xác nhận đã hết hạn"
        )

    # so sánh mã xác thực
    if verification_code["code"] != data.code:
        raise HTTPException(
            status_code=400,
            detail="Mã xác nhận không chính xác"
        )

    # Mã xác thực hợp lệ, xóa mã xác thực cũ
    await verification_codes_collection.delete_many({"email": data.email})

    # Cập nhật trạng thái xác thực tài khoản
    await users_collection.update_one({"email": data.email}, {"$set": {"confirmed": True}})
    
    return {"status": 200, "message": "Mã xác thực hợp lệ"}

@router.post("/login")
async def login(user: LoginRequest):
    """
    Đăng nhập tài khoản
    """
    # Tìm user bằng email
    db_user = await users_collection.find_one({"email": user.email})
    
    # print("mk user:", db_user["password_hash"])
    # print("mk gui len:", user.password)
    # Kiểm tra tài khoản có tồn tại không
    if not db_user:
        raise HTTPException(
            status_code=404, 
            detail="Email chưa đăng ký tài khoản"
        )
    
    # Kiểm tra tài khoản có bị revoke không
    if db_user.get("isRevoked", False):
        raise HTTPException(
            status_code=403,
            detail="Tài khoản đã bị khóa"
        )
    
    # Kiểm tra password (sử dụng password_hash thay vì password)
    if not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(
            status_code=401, 
            detail="Sai tài khoản hoặc mật khẩu"
        )


    # Kiểm tra tài khoản có xác thực không
    if not db_user.get("confirmed", False):
        raise HTTPException(
            status_code=403,
            detail="Tài khoản chưa xác thực"
        )

    # Tạo user session
    session_id = await create_user_session(db_user["id"], "local")
    
    # Tạo access token với session_id
    access_token, created_at, expires_at = create_access_token(
        data={
            "sub": db_user["email"],
            "user_id": db_user["id"],
            "name": db_user["name"],
            "role": db_user["role"],
            "session_id": session_id
        }
    )
    
    print("access_token: ", access_token)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "message": "Đăng nhập thành công",
        "session_id": session_id,
        "user": {
            "id": db_user["id"],
            "email": db_user["email"],
            "name": db_user["name"],
            "role": db_user["role"],
            "confirmed": db_user["confirmed"]
        }
    }

@router.post("/forgot-password")
async def forgot_password(data: SendVerificationCode):
    """
    Quên mật khẩu
    """
    # Kiểm tra email có tồn tại không
    db_user = await users_collection.find_one({"email": data.email})

    if not db_user:
        raise HTTPException(status_code=404, detail="Email chưa đăng ký tài khoản")
    
    token, created_at, expires_at = create_access_token(
        data={
            "sub": db_user["email"],
            "user_id": db_user["id"],
        }
    )

    expires_at_1 = datetime.now(timezone.utc) + timedelta(minutes=VERIFICATION_CODE_EXPIRE_MINUTES)
    expires_at_vn = expires_at_1.astimezone(timezone(timedelta(hours=7)))
    
    reset_password_url = f"{FRONTEND_URL}/reset-password?token={token}"
    print("reset_password_url:", reset_password_url)
    mail_service.send_forgot_password_email(data.email, token, expires_at_vn, reset_password_url)

    return {"status": 200, "message": "Email đã được gửi đến"}

# Xem token hợp lý không
@router.post("/check-token")
async def check_token(token: VerifyResetToken):
    """
    Kiểm tra token còn hạn không và trả về payload
    """
    payload = await verify_access_token(token.token)
    # print("voddaayyyyyyyyyy", payload)
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Token không hợp lệ hoặc đã hết hạn"
        )
    # Nếu decode thành công, token còn hạn
    return {
        "status": 200,
        "message": "Token còn hạn",
        "payload": payload
    }

@router.post("/verify-token")
async def verify_token(user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status": 200, "message": "Token hợp lệ"}

@router.post("/logout")
async def logout(user: dict = Depends(get_current_user)):
    """
    Đăng xuất tài khoản - cập nhật user session
    """
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Lấy session_id từ token
    session_id = user.get("session_id")
    
    if session_id:
        # Cập nhật session khi logout
        success = await end_user_session(session_id)
        if success:
            return {
                "status": 200,
                "message": "Đăng xuất thành công",
                "session_id": session_id
            }
        else:
            return {
                "status": 200,
                "message": "Đăng xuất thành công (session không tìm thấy)",
                "session_id": session_id
            }
    else:
        return {
            "status": 200,
            "message": "Đăng xuất thành công (không có session)",
        }

# Đặt lại mật khẩu
@router.post("/reset-password")
async def reset_password(data: ResetPassword):
    """
    Đặt lại mật khẩu
    """
    payload = await verify_access_token(data.token)
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Token không hợp lệ hoặc đã hết hạn"
        )
    
    try:
    # Cập nhật lại mật khẩu cho user
        await users_collection.update_one({"id": payload["user_id"]}, {"$set": {"password_hash": hash_password(data.new_password)}})
        return {"status": 200, "message": "Mật khẩu đã được đặt lại thành công"}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="Lỗi khi đặt lại mật khẩu"
        )



from pydantic import BaseModel, EmailStr
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import requests

class FacebookLoginRequest(BaseModel):
    token: str
class GoogleLoginRequest(BaseModel):
    token: str

@router.post("/google-login")
async def google_login(req: GoogleLoginRequest):
    try:

        gg_url=f"https://www.googleapis.com/oauth2/v1/userinfo"
        response = requests.get(
        gg_url,
        params={"alt": "json"},
        headers={"Authorization": f"Bearer {req.token}"}
    )
        response = response.json()
        print("response:", response)
        # Kiểm tra user có tồn tại không
        db_user = await users_collection.find_one({"provider_id": response["id"], "auth_provider": "google"})
        # user đn bằng tài khoản google lần đầu -> tạo tk
        if not db_user:
            user_data = {
                "id": response["id"],
                "email": None,
                "name": response.get("name", ""),
                "password_hash": None,
                "auth_provider": "google",
                "provider_id": response["id"],
                "isRevoked": False,
                "confirmed": True,
                "role": "user",
                "created_at": datetime.now(timezone.utc)
            }

            result = await users_collection.insert_one(user_data)

            if not result:
                raise HTTPException(status_code=400, detail="Lỗi khi tạo tài khoản")

            # Tạo user session cho Google login
            session_id = await create_user_session(user_data["id"], "google")

            access_token, created_at, expires_at = create_access_token(
                data={
                    "sub": user_data["id"],
                    "user_id": user_data["id"],
                    "name": user_data["name"],
                    "role": user_data["role"],
                    "auth_provider": user_data["auth_provider"],
                    "session_id": session_id
                }
            )
            print("access_token: ", access_token)
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "message": "Đăng nhập thành công",
                "session_id": session_id,
                "user": {
                    "id": user_data["id"],
                    "name": user_data["name"],
                    "role": user_data["role"],
                }
            }
        else:
            # Kiểm tra tài khoản có bị revoke không
            if db_user.get("isRevoked", False):
                raise HTTPException(
                    status_code=403,
                    detail="Tài khoản đã bị khóa"
                )

            # Tạo user session cho Google login existing user
            session_id = await create_user_session(db_user["id"], "google")
            
            access_token, created_at, expires_at = create_access_token(
                data={
                    "sub": db_user["id"],
                    "user_id": db_user["id"],
                    "name": db_user["name"],
                    "role": db_user["role"],
                    "auth_provider": db_user["auth_provider"],
                    "session_id": session_id
                }
            )
            print("access_token: ", access_token)
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "message": "Đăng nhập thành công",
                "session_id": session_id,
                "user": {
                    "id": db_user["id"],
                    "name": db_user["name"],
                    "role": db_user["role"],
                }
            }
    except ValueError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ")


@router.post("/facebook-login")
async def facebook_login(req: FacebookLoginRequest):
    try:
        # print("req.access_token:", req.token)
        fb_url = f"https://graph.facebook.com/me?fields=id,name,email&access_token={req.token}"
        response = requests.get(fb_url)
        response = response.json()
        # print("response:", response)
        db_user = await users_collection.find_one({"provider_id": response["id"], "auth_provider": "facebook"})
        # user đn bằng tài khoản facebook lần đầu -> tạo tk
        if not db_user:
            user_data = {
                "id": response["id"],
                "email": None,
                "name": response.get("name", ""),
                "password_hash": None,
                "auth_provider": "facebook",
                "provider_id": response["id"],
                "isRevoked": False,
                "confirmed": True,
                "role": "user",
                "created_at": datetime.now(timezone.utc)
            }

            result = await users_collection.insert_one(user_data)

            if not result:
                raise HTTPException(status_code=400, detail="Lỗi khi tạo tài khoản")

            # Tạo user session cho Facebook login
            session_id = await create_user_session(user_data["id"], "facebook")

            access_token, created_at, expires_at = create_access_token(
                data={
                    "sub": user_data["id"],
                    "user_id": user_data["id"],
                    "name": user_data["name"],
                    "role": user_data["role"],
                    "auth_provider": user_data["auth_provider"],
                    "session_id": session_id
                }
            )
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "message": "Đăng nhập thành công",
                "session_id": session_id,
                "user": {
                    "id": user_data["id"],
                    "name": user_data["name"],
                    "role": user_data["role"],
                }
            }
        else:
            # Kiểm tra tài khoản có bị revoke không
            if db_user.get("isRevoked", False):
                raise HTTPException(
                    status_code=403,
                    detail="Tài khoản đã bị khóa"
                )

            # Tạo user session cho Facebook login existing user
            session_id = await create_user_session(db_user["id"], "facebook")
            
            access_token, created_at, expires_at = create_access_token(
                data={
                    "sub": db_user["id"],
                    "user_id": db_user["id"],
                    "name": db_user["name"],
                    "role": db_user["role"],
                    "auth_provider": db_user["auth_provider"],
                    "session_id": session_id
                }
            )
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "message": "Đăng nhập thành công",
                "session_id": session_id,
                "user": {
                    "id": db_user["id"],
                    "name": db_user["name"],
                    "role": db_user["role"],
                }
            }
    except ValueError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ")

@router.post("/google-register")
async def google_register(req: GoogleLoginRequest):
    try:
        gg_url=GOOGLE_AUTH_URL
        response = requests.get(
        gg_url,
        params={"alt": "json"},
        headers={"Authorization": f"Bearer {req.token}"}
    )
        response = response.json()
        print("response:", response)
        # Kiểm tra user có tồn tại không
        existing_user = await users_collection.find_one({"provider_id": response["id"], "auth_provider": "google"})
        # user đn bằng tài khoản google lần đầu -> tạo tk
        if existing_user:
            raise HTTPException(
                status_code=400, 
                detail="Tài khoản Google đã được đăng ký"
            ) 
        else:
            user_data = {
                "id": response["id"],
                "email": None,
                "name": response.get("name", ""),
                "password_hash": None,
                "auth_provider": "google",
                "provider_id": response["id"],
                "isRevoked": False,
                "confirmed": True,
                "role": "user",
                "created_at": datetime.now(timezone.utc)
            }
            result = await users_collection.insert_one(user_data)
            if not result:
                raise HTTPException(status_code=400, detail="Lỗi khi tạo tài khoản")


            return {
                "status_code": 201,
                "message": "Đăng ký tài khoản thành công",
                "data": {
                    "user_id": user_data["id"],
                    "name": user_data["name"],
                    "role": "user",
                    "created_at": user_data["created_at"]
                }
            }
    except ValueError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ")


@router.post("/facebook-register")
async def facebook_register(req: FacebookLoginRequest):
    try:
        # print("req.access_token:", req.token)
        fb_url = f"https://graph.facebook.com/me?fields=id,name,email&access_token={req.token}"
        response = requests.get(fb_url)
        response = response.json()
        # Kiểm tra user có tồn tại không
        existing_user = await users_collection.find_one({"provider_id": response["id"], "auth_provider": "facebook"})
        # user đn bằng tài khoản facebook lần đầu -> tạo tk
        if existing_user:
            raise HTTPException(
                status_code=400, 
                detail="Tài khoản facebook đã được đăng ký"
            ) 
        else:
            user_data = {
                "id": response["id"],
                "email": None,
                "name": response.get("name", ""),
                "password_hash": None,
                "auth_provider": "facebook",
                "provider_id": response["id"],
                "isRevoked": False,
                "confirmed": True,
                "role": "user",
                "created_at": datetime.now(timezone.utc)
            }
            result = await users_collection.insert_one(user_data)
            if not result:
                raise HTTPException(status_code=400, detail="Lỗi khi tạo tài khoản")


            return {
                "status_code": 201,
                "message": "Đăng ký tài khoản thành công",
                "data": {
                    "user_id": user_data["id"],
                    "name": user_data["name"],
                    "role": "user",
                    "created_at": user_data["created_at"]
                }
            }
    except ValueError:
        raise HTTPException(status_code=401, detail="Token không hợp lệ")



# # Verify token xem còn hạn k
# @router.get("/verify-token")
# async def verify_token(user: dict = Depends(get_current_user)):
#     # Kiểm tra xem người dùng đã đăng nhập hay chưa
#     if not user:
#         raise HTTPException(status_code=401, detail="Unauthorized")
#     return {"status": 200, "message": "Token hợp lệ"}


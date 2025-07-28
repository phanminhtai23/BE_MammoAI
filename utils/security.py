from fastapi.security import OAuth2PasswordBearer

# from utils.jwt import verify_access_token
from passlib.context import CryptContext
from utils.jwt import verify_access_token
from fastapi import Depends

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/token")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = await verify_access_token(token)
    if payload is None:
        return None
    else:
        # print("payload: ", payload)
        return payload


def check_admin_role(user: dict):
    if user["role"] == "admin":
        return True
    else:
        return False

if __name__ == "__main__":
    print(hash_password("123456"))
    print(verify_password("123456", hash_password("123456")))
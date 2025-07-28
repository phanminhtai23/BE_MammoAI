from fastapi.security import OAuth2PasswordBearer
# from utils.jwt import verify_access_token
from fastapi import Depends, HTTPException
from passlib.context import CryptContext
from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.openapi.models import OAuthFlowPassword
from utils.jwt import verify_access_token

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/users/token")


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




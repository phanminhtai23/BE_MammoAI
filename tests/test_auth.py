import pytest
from fastapi.testclient import TestClient
from main import app
from utils.security import hash_password
from utils.security import verify_password, hash_password


client = TestClient(app)


def hash_password_test():
    assert hash_password("Test1234") == "$2b$12$E0F7Q9S4nac74x1gMZcObOItf3Trra/yZmuRH.FZ//S/Bxi7d2HSW"


def verify_password_test():
    assert verify_password("123456", hash_password("123456")) == True



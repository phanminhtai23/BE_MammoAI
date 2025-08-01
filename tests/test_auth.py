import pytest
from utils.security import hash_password, verify_password
import uuid
from utils.jwt import create_access_token, verify_access_token
from datetime import datetime  # Thêm import này

def test_hash_password():
    """Test hash password function"""
    hashed = hash_password("Test1234")
    assert hashed != "Test1234"  # Should be hashed
    assert len(hashed) > 10  # Should be long enough


def test_verify_password():
    """Test verify password function"""
    password = "123456"
    hashed = hash_password(password)
    
    # Should verify correctly
    assert verify_password(password, hashed) == True
    
    # Should fail with wrong password
    assert verify_password("wrong_password", hashed) == False

def test_jwt_token():
    """Test JWT token creation and verification"""
    user_id = str(uuid.uuid4())
    user_data = {
        "id": user_id,
        "email": "test@example.com",
        "name": "Test User"
    }
    
    # Test create token - function returns tuple (token, created_at, expires_at)
    token_result = create_access_token(user_data)
    assert isinstance(token_result, tuple)
    assert len(token_result) == 3
    
    token, created_at, expires_at = token_result
    
    # Test token string
    assert isinstance(token, str)
    assert len(token) > 50
    
    # Test timestamps
    assert isinstance(created_at, datetime)
    assert isinstance(expires_at, datetime)
    assert expires_at > created_at

@pytest.mark.asyncio
async def test_verify_jwt_token():
    """Test verify JWT token"""
    user_id = str(uuid.uuid4())
    user_data = {
        "id": user_id,
        "email": "test@example.com",
        "name": "Test User"
    }
    
    # Create token
    token_result = create_access_token(user_data)
    token = token_result[0]  # Lấy token string
    
    # Test verify token
    payload = await verify_access_token(token)
    assert payload is not None
    assert payload.get("id") == user_id
    assert payload.get("email") == user_data["email"]

@pytest.mark.asyncio
async def test_jwt_token_invalid():
    """Test JWT token với token không hợp lệ"""
    invalid_token = "invalid_token"
    payload = await verify_access_token(invalid_token)
    assert payload is None

def test_basic_math():
    """Basic test to ensure pytest is working"""
    assert 2 + 2 == 4
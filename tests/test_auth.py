import pytest
from utils.security import hash_password, verify_password


def test_hash_password():
    """Test hash password function"""
    hashed = hash_password("Test1234")
    assert hashed != "Test1234"  # Should be hashed
    assert len(hashed) > 20  # Should be long enough


def test_verify_password():
    """Test verify password function"""
    password = "123456"
    hashed = hash_password(password)
    
    # Should verify correctly
    assert verify_password(password, hashed) == True
    
    # Should fail with wrong password
    assert verify_password("wrong_password", hashed) == False


def test_basic_math():
    """Basic test to ensure pytest is working"""
    assert 2 + 2 == 4






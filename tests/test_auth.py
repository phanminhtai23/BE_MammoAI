import pytest
from fastapi.testclient import TestClient
from main import app
from database import users_collection, users_session_collection
from utils.security import hash_password
from datetime import datetime, timezone, timedelta
import uuid

client = TestClient(app)


def test():
    b = 2 + 2
    assert b == 4

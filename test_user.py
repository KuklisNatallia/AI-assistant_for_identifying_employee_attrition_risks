import pytest

#from Web_servis_ML.appp import UserRole
from ..models.user import User, UserRole
from ..models.models import PredictionResult
import sys
import os


def test_user_creation():
    user = User.create(email="test@test.com", username="test", password="Test1234")
    assert user.email == "test@test.com"
    assert user.username == "test"
    assert len(user.password_hash) == 64


def test_user_role():
    user = User(email="test@test.com", username="test", password_hash="hash", is_admin=True)
    assert user.role.value == "admin"

    user.is_admin = False
    assert user.role.value == "user"
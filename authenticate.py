from fastapi import Depends, HTTPException, status, Request
from .jwt_handler import get_current_user_from_token
from database.databases import get_session
from sqlmodel import Session
from passlib.context import CryptContext

# Хэширование паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


async def get_current_user(request: Request, session: Session = Depends(get_session)):
    # Токен из заголовка Authorization
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    token = auth_header.split("Bearer ")[1]

    try:
        user_data = get_current_user_from_token(token)
        return user_data
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship
from typing import List, TYPE_CHECKING
import re
from enum import Enum
from sqlmodel import Session, select
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional


if TYPE_CHECKING:
    from models.models import PredictionResult

class UserRole(Enum):
    # Роли пользователей
    USER = "user"
    ADMIN = "admin"

class User(SQLModel, table=True):
    __tablename__ = "user"
    __table_args__ = {'extend_existing': True}
    user_id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    email: str
    password_hash: str
    is_admin: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    balance: float = Field(default=0.0)

    # Связи с другими моделями
    #predictions: List["PredictionResult"] = Relationship(back_populates="user")
    #events: List["Event"] = Relationship(back_populates="creator")

    # Свойство роли пользователя
    @property
    def role(self) -> UserRole:
        return UserRole.ADMIN if self.is_admin else UserRole.USER

    @staticmethod
    def hash_password(password: str) -> str:
        import hashlib
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    @classmethod
    def create(cls, email: str, username: str, password: str) -> "User":
        return cls(
            email=email,
            username=username,
            password_hash=cls.hash_password(password),
            is_admin=False
        )

    def validate_email(self) -> None:
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not email_pattern.match(self.email):
            raise ValueError("Invalid email format")

    def validate_password(self, password: str) -> None:
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")

def get_all_users(session: Session) -> List[User]:

    try:
        users = session.exec(select(User)).all()
        return users
    except SQLAlchemyError as e:
        session.rollback()
        raise ValueError(f"Database error while fetching users: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")


def get_user_by_id(user_id: int, session: Session) -> Optional[User]:

    try:
        user = session.get(User, user_id)
        return user
    except SQLAlchemyError as e:
        session.rollback()
        raise ValueError(f"Database error while fetching user by ID: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")


def get_user_by_email(email: str, session: Session) -> Optional[User]:

    try:
        statement = select(User).where(User.email == email)
        user = session.exec(statement).first()
        return user
    except SQLAlchemyError as e:
        session.rollback()
        raise ValueError(f"Database error while fetching user by email: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")


def create_user(user_data: User, session: Session) -> User:

    try:
        # Check if user with email already exists
        existing_user = get_user_by_email(user_data.email, session)
        if existing_user:
            raise ValueError(f"User with email {user_data.email} already exists")

        session.add(user_data)
        session.commit()
        session.refresh(user_data)
        return user_data
    except SQLAlchemyError as e:
        session.rollback()
        raise ValueError(f"Database error while creating user: {str(e)}")
    except ValueError as e:
        raise
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")

def delete_user(user_id: int, session: Session) -> bool:

    try:
        user = session.get(User, user_id)
        if not user:
            return False

        session.delete(user)
        session.commit()
        return True
    except SQLAlchemyError as e:
        session.rollback()
        raise ValueError(f"Database error while deleting user: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {str(e)}")
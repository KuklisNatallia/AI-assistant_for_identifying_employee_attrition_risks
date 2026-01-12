from models.user import User
from sqlmodel import Session, select
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional


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
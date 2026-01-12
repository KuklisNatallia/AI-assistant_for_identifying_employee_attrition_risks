from .user import User
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, String
import re
#import bcrypt

class TransactionType(str, Enum):
    DEPOSIT = "deposit"
    COST_PREDICTION = "cost_prediction"
    # Можно добавить другие типы по необходимости

class Trans(SQLModel, table=True):
    __tablename__ = "trans"
    __table_args__ = {'extend_existing': True}
    transaction_id: str = Field(primary_key=True)
    user_id: int = Field(foreign_key="user.user_id")
    amount: float
    transaction_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Связь с пользователем
    # user: "User" = Relationship(back_populates="transactions")
    # user: Optional[User] = Relationship(back_populates="transactions")
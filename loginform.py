from fastapi import Request, HTTPException, status
from typing import Optional, List
from pydantic import BaseModel

class LoginFormData(BaseModel):
    username: str
    password: str

class LoginForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.errors: List[str] = []
        self.username: Optional[str] = None
        self.password: Optional[str] = None

    async def load_data(self):
        form = await self.request.form()
        self.username = form.get("username")
        self.password = form.get("password")

    async def is_valid(self):
        self.errors = []

        if not self.username:
            self.errors.append("Email is required")
        elif "@" not in self.username:
            self.errors.append("Invalid email format")

        if not self.password:
            self.errors.append("Password is required")
        elif len(self.password) < 8:
            self.errors.append("Password must be at least 8 characters")

        return len(self.errors) == 0

    def get_errors(self) -> List[str]:
        return self.errors

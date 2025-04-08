from fastapi import HTTPException

from application.api.models.login_request import LoginRequest
from application.api.repository import user_repository

async def login_user(user: LoginRequest) -> str:
    found = await user_repository.get_user_by_email(user.email)
    if not found or found["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return "User logged in"
from http.client import HTTPException

from application.api.models.RegisterRequest import RegisterRequest
from application.api.repository import user_repository

def register_service():
    return register_user

async def register_user(user: RegisterRequest):
    existing = await user_repository.get_user_by_username(user.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    return await user_repository.create_user(user)
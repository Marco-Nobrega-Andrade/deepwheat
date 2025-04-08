from http.client import HTTPException

from application.api.models.register_request import RegisterRequest
from application.api.repository import user_repository

def register_service():
    return register_user

async def register_user(user: RegisterRequest):
    existing = await user_repository.get_user_by_email(user.email)
    if existing:
        raise HTTPException(status_code=400, detail="User already registered")
    return await user_repository.create_user(user)
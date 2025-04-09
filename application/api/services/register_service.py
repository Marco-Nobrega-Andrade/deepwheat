from fastapi import HTTPException
from application.api.models.register_request import RegisterRequest
from application.api.repository import user_repository
from application.api.utils.security_utils import hash_password

async def register_user(user: RegisterRequest):
    existing = await user_repository.get_user_by_email(user.email)
    if existing:
        raise HTTPException(status_code=400, detail="User already registered")

    user.password = hash_password(user.password)
    return await user_repository.create_user(user)

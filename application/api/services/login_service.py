from http.client import HTTPException

from application.api.models import LoginRequest
from application.api.repository import user_repository


def login_service():
    return login_user

async def login_user(user: LoginRequest):
    found = await user_repository.get_user_by_username(user.username)
    if not found or found["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return { "id": found["id"], "username": found["username"] }

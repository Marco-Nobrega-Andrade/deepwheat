from application.api.infrastructure.database_connection import database
from application.api.models.RegisterRequest import RegisterRequest
from application.api.schemas.users_schema import users


async def create_user(user: RegisterRequest):
    query = users.insert().values(username=user.username, password=user.password)
    user_id = await database.execute(query)
    return { "id": user_id, "username": user.username }

async def get_user_by_username(username: str):
    query = users.select().where(users.c.username == username)
    return await database.fetch_one(query)

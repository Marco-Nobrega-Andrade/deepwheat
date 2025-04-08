from application.api.infrastructure.database_connection import database
from application.api.models.register_request import RegisterRequest
from application.api.schemas.users_schema import users


async def create_user(user: RegisterRequest):
    query = users.insert().values(
        email=user.email,
        name=user.name,
        password=user.password
    )
    user_id = await database.execute(query)
    return { "id": user_id, "name": user.name }

async def get_user_by_email(user_email: str):
    query = users.select().where(users.c.email == user_email)
    return await database.fetch_one(query)

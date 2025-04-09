from application.api.infrastructure.database_connection import database
from application.api.schemas.user_configurations_schema import user_configurations
from application.api.models.user_configurations_model import UserConfigurationsModel


async def create_user_configuration(area: float, plantation_cost: float, expected_profit: float, expected_yield: float) -> UserConfigurationsModel:
    query = user_configurations.insert().values(
        area=area,
        plantation_cost=plantation_cost,
        expected_profit=expected_profit,
        expected_yield=expected_yield
    )
    record_id = await database.execute(query)
    return UserConfigurationsModel(
        id=record_id,
        area=area,
        plantation_cost=plantation_cost,
        expected_profit=expected_profit,
        expected_yield=expected_yield
    )


async def get_user_configuration_by_id(config_id: int) -> UserConfigurationsModel | None:
    query = user_configurations.select().where(user_configurations.c.id == config_id)
    row = await database.fetch_one(query)
    return UserConfigurationsModel(**row) if row else None


async def get_all_user_configurations() -> list[UserConfigurationsModel]:
    query = user_configurations.select()
    rows = await database.fetch_all(query)
    return [UserConfigurationsModel(**row) for row in rows]


async def update_user_configuration(config_id: int, area: float, plantation_cost: float, expected_profit: float, expected_yield: float) -> UserConfigurationsModel:
    query = user_configurations.update().where(user_configurations.c.id == config_id).values(
        area=area,
        plantation_cost=plantation_cost,
        expected_profit=expected_profit,
        expected_yield=expected_yield
    )
    await database.execute(query)
    return UserConfigurationsModel(
        id=config_id,
        area=area,
        plantation_cost=plantation_cost,
        expected_profit=expected_profit,
        expected_yield=expected_yield
    )


async def delete_user_configuration(config_id: int) -> dict:
    query = user_configurations.delete().where(user_configurations.c.id == config_id)
    await database.execute(query)
    return {"message": f"User configuration with id {config_id} has been deleted"}

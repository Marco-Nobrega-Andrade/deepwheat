from typing import List, Optional
from application.api.models.user_configurations_model import UserConfigurationsModel
from application.api.models.user_configurations_request import UserConfigurationsRequest
from application.api.repository import user_configurations_repository


async def create_configuration(user_id: int, config: UserConfigurationsRequest) -> UserConfigurationsModel:
    return await user_configurations_repository.create_user_configuration(
        user_id=user_id,
        area=config.area,
        plantation_cost=config.plantation_cost,
        expected_profit=config.expected_profit,
        expected_yield=config.expected_yield
    )


async def get_configuration_by_user_id(user_id: int) -> Optional[UserConfigurationsModel]:
    return await user_configurations_repository.get_configuration_by_user_id(user_id)


async def get_all_configurations(user_id: int) -> List[UserConfigurationsModel]:
    return await user_configurations_repository.get_all_user_configurations(user_id)


async def update_configuration(user_id: int, config_id: int, config: UserConfigurationsRequest) -> UserConfigurationsModel:
    return await user_configurations_repository.update_user_configuration(
        config_id=config_id,
        user_id=user_id,
        area=config.area,
        plantation_cost=config.plantation_cost,
        expected_profit=config.expected_profit,
        expected_yield=config.expected_yield
    )


async def delete_configuration(user_id: int, config_id: int) -> dict:
    return await user_configurations_repository.delete_user_configuration(config_id, user_id)

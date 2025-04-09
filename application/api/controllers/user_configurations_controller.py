from fastapi import APIRouter, Query
from application.api.services import user_configuration_service
from application.api.models.user_configurations_model import UserConfigurationsModel

router = APIRouter(tags=["User Configurations"])

@router.post("/user/configurations", response_model=UserConfigurationsModel)
async def create_configuration(
    config: UserConfigurationsModel,
    user_id: int = Query(..., description="User ID from login")
):
    return await user_configuration_service.create_configuration(user_id, config)

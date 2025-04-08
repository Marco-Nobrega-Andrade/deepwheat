from fastapi import APIRouter, Depends

from application.api.models.LoginRequest import LoginRequest
from application.api.services import login_service

router = APIRouter(tags=["Login"])

@router.post("/login")
async def request_login(request: LoginRequest, service: callable = Depends(login_service)): return service(request)

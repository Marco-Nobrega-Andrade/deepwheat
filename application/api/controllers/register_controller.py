from fastapi import APIRouter, Depends

from application.api.models.RegisterRequest import RegisterRequest
from application.api.services import register_service

router = APIRouter(tags=["Register"])

@router.post("/register")
async def request_register(request:  RegisterRequest, service: callable = Depends(register_service)): return service(request)
from fastapi import APIRouter, Depends, HTTPException
from typing import List

from application.api.services import report_service
from application.api.utils.token_utils import get_current_user_id_from_token


router = APIRouter(tags=["Report Generation"])

from pydantic import BaseModel

class ReportRequest(BaseModel):
    image: str

@router.post("/report")
async def generate_report(request: ReportRequest, user_id: int = Depends(get_current_user_id_from_token)):
    return await report_service.generate_report(request.image, user_id)

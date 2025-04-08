
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from application.api.controllers.login_controller import router as login_router
from application.api.controllers.register_controller import router as register_router

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(login_router)
app.include_router(register_router)

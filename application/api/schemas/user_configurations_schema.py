from sqlalchemy import Table, Column, Integer, Float
from application.api.infrastructure.database_connection import metadata

user_configurations = Table(
    "user_configurations",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("area", Float, nullable=False),
    Column("plantation_cost", Float, nullable=False),
    Column("expected_profit", Float, nullable=False),
    Column("expected_yield", Float, nullable=False)
)
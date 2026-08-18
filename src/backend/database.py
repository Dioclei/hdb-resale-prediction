import os
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy import URL, create_engine
from sqlalchemy import DateTime, String, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB

load_dotenv()  # reads .env into environment variables

url_object = URL.create(
    "postgresql+psycopg2",
    username=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    host=os.environ["DB_HOST"],
    port=int(os.environ["DB_PORT"]),
    database=os.environ["DB_NAME"],
)

class Base(DeclarativeBase):
    pass

# API requests
class ApiRequest(Base):
    __tablename__ = "api_request"

    request_id: Mapped[int] = mapped_column(primary_key=True)
    inference: Mapped[Optional[Inference]] = relationship(back_populates="api_request")

# Model Inference input / output
class Inference(Base):
    __tablename__ = "inference"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model: Mapped[str] = mapped_column(String)
    input: Mapped[dict] = mapped_column(JSONB)
    output: Mapped[dict] = mapped_column(JSONB)
    api_request_id: Mapped[int] = mapped_column(ForeignKey("api_request.request_id"))
    api_request: Mapped[ApiRequest] = relationship(back_populates="inference")

# Application logs
class Log(Base):
    __tablename__ = "log"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    message: Mapped[str] = mapped_column(String)
    extra: Mapped[dict] = mapped_column(JSONB)

def setup_db():
    engine = create_engine(url_object, echo=True)
    Base.metadata.create_all(engine)

def get_url_string():
    """Returns full URL string, including password"""
    return url_object.render_as_string(hide_password=False)
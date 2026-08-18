import os
import functools
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy import URL, Engine, create_engine
from sqlalchemy import DateTime, String, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB

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

class Database():
    def __init__(self):
        self.engine: Engine | None = None
        self.url_object: URL | None = None

    @property
    def engine_(self) -> Engine:
        if self.engine is None:
            raise RuntimeError("Database not initialized, call setup_db() first")
        return self.engine

    def get_url_object(self) -> URL:
        if not self.url_object:
            load_dotenv() # reads .env into os.environ environment variables
            self.url_object = URL.create(
                "postgresql+psycopg2",
                username=os.environ["DB_USER"],
                password=os.environ["DB_PASSWORD"],
                host=os.environ["DB_HOST"],
                port=int(os.environ["DB_PORT"]),
                database=os.environ["DB_NAME"],
            )
        return self.url_object

    def get_url_string(self) -> str:
        """Returns full URL string, including password"""
        return self.get_url_object().render_as_string(hide_password=False)

    def setup_db(self) -> None:
        """Initializes database engine"""
        if not self.engine:
            self.engine = create_engine(self.get_url_object(), echo=True)

    def shutdown(self) -> None:
        """Shutdowns database engine and cleans up all database connections"""
        if self.engine:
            self.engine.dispose()

    @staticmethod
    def db_query(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if not self.engine:
                raise RuntimeError("Database not initialized, call setup_db() first")
            return fn(self, *args, **kwargs)
        return wrapper

    # example query
    @db_query
    def get_data(self, someparam):
        # do something
        print(f"hello world! {someparam}")

db = Database()
import os
import functools
from typing import Optional, Generator
from dotenv import load_dotenv
from datetime import datetime
from sqlalchemy import URL, Engine, create_engine
from sqlalchemy import DateTime, String, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session
from sqlalchemy.dialects.postgresql import JSONB

class Base(DeclarativeBase):
    pass

# API requests
class ApiRequest(Base):
    __tablename__ = "api_request"

    request_id: Mapped[str] = mapped_column(String(12), primary_key=True)
    inference: Mapped[Optional[Inference]] = relationship(back_populates="api_request")

# Model Inference input / output
class Inference(Base):
    __tablename__ = "inference"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    model: Mapped[str] = mapped_column(String)
    input: Mapped[dict] = mapped_column(JSONB)
    output: Mapped[dict] = mapped_column(JSONB)
    api_request_id: Mapped[str] = mapped_column(ForeignKey("api_request.request_id"))
    api_request: Mapped[ApiRequest] = relationship(back_populates="inference")

# Application logs
class Log(Base):
    __tablename__ = "log"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    message: Mapped[str] = mapped_column(String)
    extra: Mapped[dict] = mapped_column(JSONB)

class Database:
    """Owns the engine, URL config, and session factory for one database connection."""
 
    def __init__(self) -> None:
        self._url_object: URL | None = None
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None

    def get_url_object(self) -> URL:
        if self._url_object is None:
            load_dotenv()  # reads .env into os.environ environment variables
            self._url_object = URL.create(
                "postgresql+psycopg",
                username=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"],
                host=os.environ["POSTGRES_HOST"],
                port=int(os.environ["POSTGRES_PORT"]),
                database=os.environ["POSTGRES_DB"],
            )
        return self._url_object

    def get_url_string(self, hide_password=True) -> str:
        """Returns full URL string, including password. Debug use only — do not log."""
        return self.get_url_object().render_as_string(hide_password=hide_password)

    def setup_db(self) -> None:
        """Initializes the engine and session factory. Safe to call more than once."""
        if self._engine is not None:
            return
        self._engine = create_engine(self.get_url_object(), echo=True)
        self._session_factory = sessionmaker(
            bind=self._engine, autoflush=False, autocommit=False
        )

    def shutdown(self) -> None:
        """Shuts down the engine and cleans up all database connections."""
        if self._engine is not None:
            self._engine.dispose()

    def new_session(self) -> Session:
        if self._session_factory is None:
            raise RuntimeError("Database not initialized, call setup_db() first")
        return self._session_factory()


# Single shared instance used by the whole app.
database = Database()

# Module-level export to get a session, for instance in FastAPI routes
def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a Session scoped to one request."""
    session = database.new_session()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
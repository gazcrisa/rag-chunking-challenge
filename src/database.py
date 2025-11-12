from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.settings import get_settings

settings = get_settings()
engine = create_engine(settings.build_sync_sqlalchemy_url())
SessionLocal = sessionmaker(bind=engine)


def get_db():
    """Generator-style dependency, e.g., for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session():
    """
    Context manager for standalone scripts.

    Example:
        with db_session() as session:
            session.add(obj)
            session.commit()
    """
    db_gen = get_db()
    session = next(db_gen)
    try:
        yield session
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass

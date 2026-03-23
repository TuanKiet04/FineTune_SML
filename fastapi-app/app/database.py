from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

#DATABASE_URL = os.getenv(
#    "DATABASE_URL",
#    "postgresql://${PG_USER}:${PG_PASS}@postgres:5432/optimize"
#)
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency để get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
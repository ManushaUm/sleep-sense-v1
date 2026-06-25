import os
import urllib.parse
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

def encode_db_url(db_url: str) -> str:
    prefix = "postgresql://"
    if db_url.startswith(prefix):
        main_part = db_url[len(prefix):]
        if "?" in main_part:
            auth_path, query = main_part.split("?", 1)
            query_str = "?" + query
        else:
            auth_path = main_part
            query_str = ""
        if "/" in auth_path:
            authority, path = auth_path.split("/", 1)
            path_str = "/" + path
        else:
            authority = auth_path
            path_str = ""
        if "@" in authority:
            creds, host = authority.rsplit("@", 1)
            if ":" in creds:
                user, password = creds.split(":", 1)
                encoded_password = urllib.parse.quote_plus(password)
                return f"{prefix}{user}:{encoded_password}@{host}{path_str}{query_str}"
    return db_url

# Detect if we are running in production using DATABASE_URL env var
raw_db_url = os.getenv("DATABASE_URL", "sqlite:///./data/sleep_study.db")
SQLALCHEMY_DATABASE_URL = encode_db_url(raw_db_url)

# SQLite needs connect_args; PostgreSQL does not
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

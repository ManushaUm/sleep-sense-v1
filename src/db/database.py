from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

REGISTRY_DIR = Path(__file__).resolve().parents[2] / "models" / "registry"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = REGISTRY_DIR / "sleepsense.db"

DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

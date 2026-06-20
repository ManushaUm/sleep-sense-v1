from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["Health"])
def health_check():
    """Liveness probe to confirm backend is running."""
    return {"status": "healthy"}

import json
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from src.db.database import get_db
from src.db import crud
from app.api import schemas, security
from google.oauth2 import id_token
from google.auth.transport import requests

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register", response_model=schemas.UserResponse)
def register_user(user_data: schemas.UserRegister, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_data.user_id)
    if db_user and db_user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already registered."
        )
    
    hashed_pwd = security.get_password_hash(user_data.password)
    # Register/update user with hashed password
    user = crud.create_or_update_user(
        db=db,
        user_id=user_data.user_id,
        psqi_pre_score=user_data.psqi_pre_score,
        psqi_post_score=user_data.psqi_post_score,
        personality_dict=user_data.personality,
        hashed_password=hashed_pwd
    )
    
    p_dict = json.loads(user.personality_json) if user.personality_json else None
    return schemas.UserResponse(
        user_id=user.user_id,
        psqi_pre_score=user.psqi_pre_score,
        psqi_post_score=user.psqi_post_score,
        personality=p_dict
    )

@router.post("/token", response_model=schemas.TokenResponse)
def login_user(login_data: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, login_data.user_id)
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user ID or password."
        )
    
    if not db_user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User has not set a password. Please register first."
        )
        
    if not security.verify_password(login_data.password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect user ID or password."
        )
        
    access_token = security.create_access_token(subject=db_user.user_id)
    return schemas.TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=db_user.user_id,
        profile_picture_url=db_user.profile_picture_url
    )

@router.post("/google-login", response_model=schemas.TokenResponse)
def google_login(payload: schemas.GoogleLoginRequest, db: Session = Depends(get_db)):
    token = payload.id_token
    GOOGLE_CLIENT_ID = "656306996421-8q1shbdao820ekjnhuodjuesmesl4fce.apps.googleusercontent.com"
    
    try:
        # Verify the ID token signature and audience
        id_info = id_token.verify_oauth2_token(token, requests.Request(), GOOGLE_CLIENT_ID)
        
        # Verify issuer
        if id_info['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
            raise ValueError('Invalid Google token issuer.')
            
        # User details
        email = id_info['email']
        picture = id_info.get('picture', None)
        user_id = email
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Google ID token: {str(e)}"
        )
        
    # Check if user already exists
    user = crud.get_user(db, user_id)
    if not user:
        # Create user automatically (registering from Google)
        user = crud.create_or_update_user(
            db=db,
            user_id=user_id,
            profile_picture_url=picture
        )
    else:
        # Update user's profile picture if it changed
        if user.profile_picture_url != picture:
            user.profile_picture_url = picture
            db.commit()
            db.refresh(user)
            
    # Generate our local JWT access token
    access_token = security.create_access_token(subject=user.user_id)
    
    return schemas.TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user.user_id,
        profile_picture_url=user.profile_picture_url
    )


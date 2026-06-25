import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.api.main import app

client = TestClient(app)

def test_auth_flow():
    # 1. Register a test user
    user_id = "test_user_auth"
    password = "securepassword123"
    
    # Clean up user if already exists from previous runs by using a unique id
    import uuid
    rand_id = f"user_{uuid.uuid4().hex[:8]}"
    
    # Register success
    r = client.post("/auth/register", json={
        "user_id": rand_id,
        "password": password,
        "psqi_pre_score": 7.0,
        "personality": {
            "extraversion": 3.0,
            "agreeableness": 3.0,
            "conscientiousness": 3.0,
            "neuroticism": 3.0,
            "openness": 3.0
        }
    })
    assert r.status_code == 200
    res = r.json()
    assert res["user_id"] == rand_id
    assert res["psqi_pre_score"] == 7.0

    # Register duplicate failure
    r_dup = client.post("/auth/register", json={
        "user_id": rand_id,
        "password": password
    })
    assert r_dup.status_code == 400

    # 2. Login success
    r_login = client.post("/auth/token", json={
        "user_id": rand_id,
        "password": password
    })
    assert r_login.status_code == 200
    token_res = r_login.json()
    assert "access_token" in token_res
    assert token_res["token_type"] == "bearer"
    assert token_res["user_id"] == rand_id
    
    token = token_res["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Login incorrect password failure
    r_wrong_pwd = client.post("/auth/token", json={
        "user_id": rand_id,
        "password": "wrongpassword"
    })
    assert r_wrong_pwd.status_code == 401

    # Login unregistered user failure
    r_no_user = client.post("/auth/token", json={
        "user_id": "non_existent_user_999",
        "password": password
    })
    assert r_no_user.status_code == 401

    # 3. Test protected endpoints
    # Unauthorized request to history
    r_hist_unauth = client.get(f"/history/{rand_id}")
    assert r_hist_unauth.status_code == 401 # No token provided
    
    # Authorized request to history
    r_hist_auth = client.get(f"/history/{rand_id}", headers=headers)
    assert r_hist_auth.status_code == 200
    assert isinstance(r_hist_auth.json(), list)

    # Request with another user's token (forbidden)
    r_login_other = client.post("/auth/register", json={
        "user_id": rand_id + "_other",
        "password": password
    })
    assert r_login_other.status_code == 200
    token_other = client.post("/auth/token", json={
        "user_id": rand_id + "_other",
        "password": password
    }).json()["access_token"]
    
    r_hist_forbidden = client.get(f"/history/{rand_id}", headers={"Authorization": f"Bearer {token_other}"})
    assert r_hist_forbidden.status_code == 403

from unittest.mock import patch

def test_google_login():
    # Mock verify_oauth2_token
    with patch("google.oauth2.id_token.verify_oauth2_token") as mock_verify:
        mock_verify.return_value = {
            'iss': 'accounts.google.com',
            'email': 'google_test_user@gmail.com',
            'picture': 'https://example.com/google_photo.jpg'
        }
        
        # Call the endpoint
        r = client.post("/auth/google-login", json={"id_token": "fake_google_token"})
        assert r.status_code == 200
        res = r.json()
        assert "access_token" in res
        assert res["token_type"] == "bearer"
        assert res["user_id"] == "google_test_user@gmail.com"
        assert res["profile_picture_url"] == "https://example.com/google_photo.jpg"
        
        # Verify user registration and picture in DB via /users listing
        r_users = client.get("/users")
        assert r_users.status_code == 200
        users_list = r_users.json()
        matched_user = next((u for u in users_list if u["user_id"] == "google_test_user@gmail.com"), None)
        assert matched_user is not None
        assert matched_user["profile_picture_url"] == "https://example.com/google_photo.jpg"


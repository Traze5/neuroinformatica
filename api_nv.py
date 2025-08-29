# api_nv.py
import os, time, httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

OG_TOKEN_URL = os.environ["OG_TOKEN_URL"]
OG_CLIENT_ID = os.environ["OG_CLIENT_ID"]
OG_CLIENT_SECRET = os.environ["OG_CLIENT_SECRET"]
OG_SCOPE = os.getenv("OG_SCOPE", "number-verification")
OG_BASE_URL = os.environ["OG_BASE_URL"]

_token_cache = {"access_token": None, "exp": 0}

async def get_access_token():
    # cache sencillo
    if _token_cache["access_token"] and _token_cache["exp"] - time.time() > 30:
        return _token_cache["access_token"]
    async with httpx.AsyncClient(timeout=15) as client:
        data = {
            "grant_type": "client_credentials",
            "client_id": OG_CLIENT_ID,
            "client_secret": OG_CLIENT_SECRET,
            "scope": OG_SCOPE
        }
        r = await client.post(OG_TOKEN_URL, data=data)
        if r.status_code != 200:
            raise HTTPException(500, f"OIDC token error: {r.text}")
        tok = r.json()
        _token_cache["access_token"] = tok["access_token"]
        _token_cache["exp"] = time.time() + tok.get("expires_in", 300)
        return tok["access_token"]

class VerifyIn(BaseModel):
    phoneNumber: str  # E.164, p.ej. "+346XXXXXXXX"

@app.post("/nv/verify")
async def nv_verify(body: VerifyIn):
    token = await get_access_token()
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(
            f"{OG_BASE_URL}/verify",
            json={"phoneNumber": body.phoneNumber},
            headers={"Authorization": f"Bearer {token}"}
        )
    if r.status_code not in (200, 400, 401, 403):
        raise HTTPException(r.status_code, r.text)
    return r.json()

@app.get("/nv/device-phone-number")
async def nv_device_phone_number():
    token = await get_access_token()
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            f"{OG_BASE_URL}/device-phone-number",
            headers={"Authorization": f"Bearer {token}"}
        )
    if r.status_code != 200:
        raise HTTPException(r.status_code, r.text)
    return r.json()

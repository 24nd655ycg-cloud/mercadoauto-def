"""MercadoAuto - Main FastAPI server.
Multi-tenant SaaS for Brazilian companies to auto-publish products on Mercado Livre.
"""
import os
import io
import uuid
import logging
import secrets
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

# Load env FIRST before importing our modules that read env at import time
from dotenv import load_dotenv
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File, Query, Header, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import pandas as pd
import re

from auth_utils import hash_password, verify_password, create_access_token, get_current_user_id
from storage_utils import init_storage, put_object, get_object, APP_NAME, MIME_TYPES
from ai_utils import generate_listing_content
import ml_utils

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

app = FastAPI(title="MercadoAuto API")
api = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Convert Pydantic 422 array errors to a single string message
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


@app.exception_handler(RequestValidationError)
async def _validation_handler(request, exc: RequestValidationError):
    errs = exc.errors()
    msg = errs[0].get("msg", "Dados inválidos") if errs else "Dados inválidos"
    if msg.startswith("Value error, "):
        msg = msg[len("Value error, "):]
    return JSONResponse(status_code=422, content={"detail": msg})


# ============== Models ==============
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class BaseDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")


class UserRegister(BaseDoc):
    email: str
    password: str
    company_name: str

    @field_validator("email")
    @classmethod
    def _v_email(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if not EMAIL_REGEX.match(v):
            raise ValueError("Email inválido")
        return v

    @field_validator("password")
    @classmethod
    def _v_pwd(cls, v: str) -> str:
        if not v or len(v) < 6:
            raise ValueError("A senha precisa ter pelo menos 6 caracteres")
        return v

    @field_validator("company_name")
    @classmethod
    def _v_company(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Informe o nome da empresa")
        return v


class UserLogin(BaseDoc):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _v_email(cls, v: str) -> str:
        return (v or "").strip().lower()


class UserOut(BaseDoc):
    id: str
    email: str
    company_name: str
    ml_connected: bool = False
    ml_user_id: Optional[str] = None
    ml_nickname: Optional[str] = None


class ProductCreate(BaseDoc):
    sku: str
    title: str
    description: str = ""
    price: float
    quantity: int = 1
    brand: str = ""
    category: str = ""
    condition: str = "new"  # new | used
    image_ids: List[str] = []
    external_image_urls: List[str] = []
    use_ai: bool = True


class ProductUpdate(BaseDoc):
    title: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    quantity: Optional[int] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    condition: Optional[str] = None
    image_ids: Optional[List[str]] = None
    external_image_urls: Optional[List[str]] = None


class ProductOut(BaseDoc):
    id: str
    sku: str
    title: str
    ai_title: Optional[str] = None
    description: str
    ai_description: Optional[str] = None
    price: float
    quantity: int
    brand: str
    category: str
    condition: str
    image_ids: List[str] = []
    external_image_urls: List[str] = []
    status: str  # draft | published | error
    ml_id: Optional[str] = None
    ml_permalink: Optional[str] = None
    ml_mock: bool = False
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class AIRequest(BaseDoc):
    title: str
    description: str = ""
    brand: str = ""
    category: str = ""


class MLConfigIn(BaseDoc):
    ml_client_id: str
    ml_client_secret: str
    ml_redirect_uri: str


# ============== Auth ==============
@api.post("/auth/register")
async def register(payload: UserRegister):
    existing = await db.users.find_one({"email": payload.email.lower()})
    if existing:
        raise HTTPException(status_code=400, detail="Email já cadastrado")
    user_id = str(uuid.uuid4())
    doc = {
        "id": user_id,
        "email": payload.email.lower(),
        "password_hash": hash_password(payload.password),
        "company_name": payload.company_name,
        "ml_connected": False,
        "ml_user_id": None,
        "ml_nickname": None,
        "ml_access_token": None,
        "ml_refresh_token": None,
        "ml_token_expires_at": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.users.insert_one(doc)
    token = create_access_token(user_id, payload.email.lower())
    return {"token": token, "user": _user_public(doc)}


@api.post("/auth/login")
async def login(payload: UserLogin):
    doc = await db.users.find_one({"email": payload.email.lower()})
    if not doc or not verify_password(payload.password, doc["password_hash"]):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")
    token = create_access_token(doc["id"], doc["email"])
    return {"token": token, "user": _user_public(doc)}


@api.get("/auth/me")
async def me(user_id: str = Depends(get_current_user_id)):
    doc = await db.users.find_one({"id": user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    return _user_public(doc)


def _user_public(doc: dict) -> dict:
    return {
        "id": doc["id"],
        "email": doc["email"],
        "company_name": doc["company_name"],
        "ml_connected": doc.get("ml_connected", False),
        "ml_user_id": doc.get("ml_user_id"),
        "ml_nickname": doc.get("ml_nickname"),
        "ml_client_id": doc.get("ml_client_id") or "",
        "ml_redirect_uri": doc.get("ml_redirect_uri") or "",
        "ml_has_secret": bool(doc.get("ml_client_secret")),
    }


# ============== File uploads (product photos) ==============
@api.post("/uploads")
async def upload_image(file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    ext = (file.filename or "").split(".")[-1].lower() if "." in (file.filename or "") else "jpg"
    if ext not in MIME_TYPES:
        raise HTTPException(status_code=400, detail="Formato de imagem não suportado")
    file_id = str(uuid.uuid4())
    path = f"{APP_NAME}/uploads/{user_id}/{file_id}.{ext}"
    data = await file.read()
    content_type = file.content_type or MIME_TYPES[ext]
    try:
        result = put_object(path, data, content_type)
    except Exception as e:
        logger.error(f"upload failed: {e}")
        raise HTTPException(status_code=500, detail="Falha ao enviar arquivo")
    doc = {
        "id": file_id,
        "user_id": user_id,
        "storage_path": result["path"],
        "original_filename": file.filename,
        "content_type": content_type,
        "size": result.get("size", len(data)),
        "is_deleted": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.files.insert_one(doc)
    return {"id": file_id, "url": f"/api/files/{file_id}", "size": doc["size"]}


@api.get("/files/{file_id}")
async def get_file(file_id: str, auth: Optional[str] = Query(None), authorization: Optional[str] = Header(None)):
    # Support token via query param for <img> tags
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
    elif auth:
        token = auth
    if not token:
        raise HTTPException(status_code=401, detail="Não autenticado")
    from auth_utils import decode_token
    payload = decode_token(token)
    user_id = payload["sub"]
    record = await db.files.find_one({"id": file_id, "user_id": user_id, "is_deleted": False})
    if not record:
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    data, content_type = get_object(record["storage_path"])
    return Response(content=data, media_type=record.get("content_type", content_type))


# ============== AI generation ==============
@api.post("/ai/generate")
async def ai_generate(payload: AIRequest, user_id: str = Depends(get_current_user_id)):
    result = await generate_listing_content(payload.title, payload.description, payload.brand, payload.category)
    return result


# ============== Products ==============
@api.post("/products", response_model=ProductOut)
async def create_product(payload: ProductCreate, user_id: str = Depends(get_current_user_id)):
    now = datetime.now(timezone.utc).isoformat()
    product_id = str(uuid.uuid4())
    ai_title = None
    ai_description = None
    if payload.use_ai:
        ai_result = await generate_listing_content(payload.title, payload.description, payload.brand, payload.category)
        ai_title = ai_result.get("title")
        ai_description = ai_result.get("description")
    doc = {
        "id": product_id,
        "user_id": user_id,
        "sku": payload.sku,
        "title": payload.title,
        "ai_title": ai_title,
        "description": payload.description,
        "ai_description": ai_description,
        "price": payload.price,
        "quantity": payload.quantity,
        "brand": payload.brand,
        "category": payload.category,
        "condition": payload.condition,
        "image_ids": payload.image_ids,
        "external_image_urls": payload.external_image_urls,
        "status": "draft",
        "ml_id": None,
        "ml_permalink": None,
        "ml_mock": False,
        "error_message": None,
        "created_at": now,
        "updated_at": now,
    }
    await db.products.insert_one(doc)
    return _product_public(doc)


@api.get("/products", response_model=List[ProductOut])
async def list_products(
    user_id: str = Depends(get_current_user_id),
    status: Optional[str] = None,
    q: Optional[str] = None,
):
    query = {"user_id": user_id}
    if status:
        query["status"] = status
    if q:
        query["$or"] = [
            {"title": {"$regex": q, "$options": "i"}},
            {"sku": {"$regex": q, "$options": "i"}},
        ]
    docs = await db.products.find(query, {"_id": 0}).sort("created_at", -1).to_list(500)
    return [_product_public(d) for d in docs]


@api.get("/products/{product_id}", response_model=ProductOut)
async def get_product(product_id: str, user_id: str = Depends(get_current_user_id)):
    doc = await db.products.find_one({"id": product_id, "user_id": user_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    return _product_public(doc)


@api.patch("/products/{product_id}", response_model=ProductOut)
async def update_product(product_id: str, payload: ProductUpdate, user_id: str = Depends(get_current_user_id)):
    doc = await db.products.find_one({"id": product_id, "user_id": user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    updates = {k: v for k, v in payload.model_dump(exclude_unset=True).items() if v is not None}
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    await db.products.update_one({"id": product_id}, {"$set": updates})
    doc.update(updates)
    return _product_public(doc)


@api.delete("/products/{product_id}")
async def delete_product(product_id: str, user_id: str = Depends(get_current_user_id)):
    res = await db.products.delete_one({"id": product_id, "user_id": user_id})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    return {"ok": True}


@api.post("/products/{product_id}/publish", response_model=ProductOut)
async def publish_product(product_id: str, user_id: str = Depends(get_current_user_id)):
    doc = await db.products.find_one({"id": product_id, "user_id": user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Produto não encontrado")
    user = await db.users.find_one({"id": user_id})
    title = doc.get("ai_title") or doc["title"]
    now = datetime.now(timezone.utc).isoformat()

    # Real ML publish path
    if ml_utils.is_configured(user) and user and user.get("ml_access_token"):
        try:
            item_payload = {
                "title": title[:60],
                "category_id": doc.get("category") or "MLB1055",
                "price": doc["price"],
                "currency_id": "BRL",
                "available_quantity": doc["quantity"],
                "buying_mode": "buy_it_now",
                "listing_type_id": "gold_special",
                "condition": doc.get("condition", "new"),
                "description": {"plain_text": doc.get("ai_description") or doc.get("description", "")},
                "pictures": [{"source": u} for u in doc.get("external_image_urls", [])],
            }
            resp = ml_utils.publish_item(user["ml_access_token"], item_payload)
            updates = {
                "status": "published",
                "ml_id": resp.get("id"),
                "ml_permalink": resp.get("permalink"),
                "ml_mock": False,
                "error_message": None,
                "updated_at": now,
            }
        except Exception as e:
            logger.error(f"ML publish failed: {e}")
            updates = {"status": "error", "error_message": str(e), "updated_at": now}
    else:
        # Mock publish - simulate for MVP demo
        result = ml_utils.mock_publish(doc)
        updates = {
            "status": "published",
            "ml_id": result["id"],
            "ml_permalink": result["permalink"],
            "ml_mock": True,
            "error_message": None,
            "updated_at": now,
        }
    await db.products.update_one({"id": product_id}, {"$set": updates})
    doc.update(updates)
    return _product_public(doc)


@api.post("/products/bulk-publish")
async def bulk_publish(product_ids: List[str], user_id: str = Depends(get_current_user_id)):
    results = []
    for pid in product_ids:
        try:
            r = await publish_product(pid, user_id)
            results.append({"id": pid, "ok": True, "status": r.status if hasattr(r, "status") else "published"})
        except Exception as e:
            results.append({"id": pid, "ok": False, "error": str(e)})
    return {"results": results}


def _product_public(doc: dict) -> dict:
    return {
        "id": doc["id"],
        "sku": doc.get("sku", ""),
        "title": doc.get("title", ""),
        "ai_title": doc.get("ai_title"),
        "description": doc.get("description", ""),
        "ai_description": doc.get("ai_description"),
        "price": doc.get("price", 0),
        "quantity": doc.get("quantity", 0),
        "brand": doc.get("brand", ""),
        "category": doc.get("category", ""),
        "condition": doc.get("condition", "new"),
        "image_ids": doc.get("image_ids", []),
        "external_image_urls": doc.get("external_image_urls", []),
        "status": doc.get("status", "draft"),
        "ml_id": doc.get("ml_id"),
        "ml_permalink": doc.get("ml_permalink"),
        "ml_mock": doc.get("ml_mock", False),
        "error_message": doc.get("error_message"),
        "created_at": doc.get("created_at", ""),
        "updated_at": doc.get("updated_at", ""),
    }


# ============== Spreadsheet import ==============
@api.post("/products/import-sheet")
async def import_sheet(file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    content = await file.read()
    name = (file.filename or "").lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao ler planilha: {e}")

    df.columns = [str(c).strip().lower() for c in df.columns]
    required = {"sku", "title", "price"}
    if not required.issubset(set(df.columns)):
        raise HTTPException(status_code=400, detail=f"Planilha deve conter colunas: {sorted(required)}. Encontradas: {list(df.columns)}")

    now = datetime.now(timezone.utc).isoformat()
    created = 0
    errors = []
    for idx, row in df.iterrows():
        try:
            def _get(col, default=""):
                v = row.get(col, default)
                if pd.isna(v):
                    return default
                return v
            image_urls = str(_get("images", "")).strip()
            urls = [u.strip() for u in image_urls.split("|") if u.strip()] if image_urls else []
            doc = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "sku": str(_get("sku")),
                "title": str(_get("title")),
                "ai_title": None,
                "description": str(_get("description", "")),
                "ai_description": None,
                "price": float(_get("price", 0)),
                "quantity": int(float(_get("quantity", 1))),
                "brand": str(_get("brand", "")),
                "category": str(_get("category", "")),
                "condition": str(_get("condition", "new")) or "new",
                "image_ids": [],
                "external_image_urls": urls,
                "status": "draft",
                "ml_id": None,
                "ml_permalink": None,
                "ml_mock": False,
                "error_message": None,
                "created_at": now,
                "updated_at": now,
            }
            await db.products.insert_one(doc)
            created += 1
        except Exception as e:
            errors.append({"row": int(idx) + 2, "error": str(e)})
    return {"created": created, "errors": errors}


# ============== External ERP Integration ==============
@api.post("/integrations/erp/products")
async def erp_ingest(products: List[ProductCreate], user_id: str = Depends(get_current_user_id)):
    """Ingest bulk products from external ERP API."""
    now = datetime.now(timezone.utc).isoformat()
    created = 0
    for p in products:
        doc = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "sku": p.sku,
            "title": p.title,
            "ai_title": None,
            "description": p.description,
            "ai_description": None,
            "price": p.price,
            "quantity": p.quantity,
            "brand": p.brand,
            "category": p.category,
            "condition": p.condition,
            "image_ids": p.image_ids,
            "external_image_urls": p.external_image_urls,
            "status": "draft",
            "ml_id": None,
            "ml_permalink": None,
            "ml_mock": False,
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }
        await db.products.insert_one(doc)
        created += 1
    return {"created": created}


# ============== Mercado Livre OAuth ==============
@api.get("/ml/status")
async def ml_status(user_id: str = Depends(get_current_user_id)):
    user = await db.users.find_one({"id": user_id})
    configured = ml_utils.is_configured(user)
    return {
        "configured": configured,
        "connected": bool(user and user.get("ml_connected")),
        "ml_nickname": user.get("ml_nickname") if user else None,
        "ml_client_id": (user.get("ml_client_id") if user else "") or "",
        "ml_redirect_uri": (user.get("ml_redirect_uri") if user else "") or "",
        "ml_has_secret": bool(user and user.get("ml_client_secret")),
        "mock_mode": not configured,
    }


@api.post("/ml/config")
async def ml_config(payload: MLConfigIn, user_id: str = Depends(get_current_user_id)):
    updates = {
        "ml_client_id": payload.ml_client_id.strip(),
        "ml_client_secret": payload.ml_client_secret.strip(),
        "ml_redirect_uri": payload.ml_redirect_uri.strip(),
    }
    await db.users.update_one({"id": user_id}, {"$set": updates})
    return {"ok": True}


@api.get("/ml/authorize")
async def ml_authorize(user_id: str = Depends(get_current_user_id)):
    user = await db.users.find_one({"id": user_id})
    if not ml_utils.is_configured(user):
        raise HTTPException(status_code=400, detail="Configure Client ID, Client Secret e Redirect URI antes de conectar")
    state = f"{user_id}:{secrets.token_urlsafe(16)}"
    await db.oauth_states.insert_one({"state": state, "user_id": user_id, "created_at": datetime.now(timezone.utc).isoformat()})
    return {"authorize_url": ml_utils.build_authorize_url(user, state)}


@api.get("/ml/callback")
async def ml_callback(code: str, state: str):
    st = await db.oauth_states.find_one({"state": state})
    if not st:
        raise HTTPException(status_code=400, detail="State inválido")
    user = await db.users.find_one({"id": st["user_id"]})
    try:
        tokens = ml_utils.exchange_code(user, code)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao trocar code: {e}")
    await db.users.update_one(
        {"id": st["user_id"]},
        {"$set": {
            "ml_connected": True,
            "ml_user_id": str(tokens.get("user_id", "")),
            "ml_access_token": tokens.get("access_token"),
            "ml_refresh_token": tokens.get("refresh_token"),
            "ml_token_expires_at": tokens.get("expires_in"),
        }},
    )
    await db.oauth_states.delete_one({"state": state})
    frontend_url = os.environ.get("REACT_APP_BACKEND_URL", "/")
    return RedirectResponse(url=f"{frontend_url}/settings?ml=connected")


@api.post("/ml/disconnect")
async def ml_disconnect(user_id: str = Depends(get_current_user_id)):
    await db.users.update_one(
        {"id": user_id},
        {"$set": {
            "ml_connected": False,
            "ml_access_token": None,
            "ml_refresh_token": None,
            "ml_user_id": None,
            "ml_nickname": None,
        }},
    )
    return {"ok": True}


@api.post("/ml/import-listings")
async def ml_import_listings(user_id: str = Depends(get_current_user_id)):
    """Busca os anúncios reais da conta do Mercado Livre conectada e os grava
    (ou atualiza) na coleção de produtos, para aparecerem no Painel e em Anúncios."""
    user = await db.users.find_one({"id": user_id})
    if not user or not user.get("ml_connected") or not user.get("ml_access_token"):
        raise HTTPException(status_code=400, detail="Conecte sua conta do Mercado Livre antes de importar os anúncios")

    try:
        listings = ml_utils.fetch_seller_listings(user["ml_access_token"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Falha ao buscar anúncios no Mercado Livre: {e}")

    imported, updated = 0, 0
    for item in listings:
        existing = await db.products.find_one({"user_id": user_id, "ml_item_id": item["ml_item_id"]})
        doc_fields = {
            "user_id": user_id,
            "sku": item["sku"],
            "title": item["title"],
            "price": item["price"],
            "quantity": item["available_quantity"],
            "status": item["status"],
            "ml_item_id": item["ml_item_id"],
            "ml_permalink": item.get("permalink"),
            "ml_thumbnail": item.get("thumbnail"),
            "source": "mercado_livre_sync",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if existing:
            await db.products.update_one({"id": existing["id"]}, {"$set": doc_fields})
            updated += 1
        else:
            doc_fields["id"] = str(uuid.uuid4())
            doc_fields["created_at"] = doc_fields["updated_at"]
            await db.products.insert_one(doc_fields)
            imported += 1

    return {"ok": True, "imported": imported, "updated": updated, "total_fetched": len(listings)}


@api.get("/products/lookup-sku")
async def lookup_sku(sku: str, brand: str = "", user_id: str = Depends(get_current_user_id)):
    """Ao digitar um SKU na tela de novo anúncio, sugere título e faixa de
    preço — primeiro olhando o próprio catálogo da empresa (se esse SKU já
    foi cadastrado/importado antes), e senão buscando anúncios reais e
    parecidos no catálogo público do Mercado Livre."""
    sku = (sku or "").strip()
    if not sku:
        return {"found": False}

    existing = await db.products.find_one({"user_id": user_id, "sku": sku})
    if existing:
        return {
            "found": True,
            "source": "own_catalog",
            "title": existing.get("title", ""),
            "description": existing.get("description", ""),
            "price": existing.get("price"),
        }

    suggestion = ml_utils.suggest_from_sku(sku, brand)
    if not suggestion.get("found"):
        return {"found": False}

    return {
        "found": True,
        "source": "mercado_livre_search",
        "title": suggestion["suggested_title"],
        "price_min": suggestion["price_min"],
        "price_max": suggestion["price_max"],
        "sample_count": suggestion["sample_count"],
    }


# ============== Dashboard stats ==============
@api.get("/stats")
async def stats(user_id: str = Depends(get_current_user_id)):
    total = await db.products.count_documents({"user_id": user_id})
    published = await db.products.count_documents({"user_id": user_id, "status": "published"})
    drafts = await db.products.count_documents({"user_id": user_id, "status": "draft"})
    errors = await db.products.count_documents({"user_id": user_id, "status": "error"})
    recent = await db.products.find({"user_id": user_id}, {"_id": 0}).sort("created_at", -1).limit(5).to_list(5)
    return {
        "total_products": total,
        "published": published,
        "drafts": drafts,
        "errors": errors,
        "recent": [_product_public(d) for d in recent],
    }


@api.get("/")
async def root():
    return {"service": "MercadoAuto API", "status": "ok"}


app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    try:
        init_storage()
    except Exception as e:
        logger.warning(f"Storage init deferred: {e}")


@app.on_event("shutdown")
async def shutdown():
    client.close()

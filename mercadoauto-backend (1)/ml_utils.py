"""Mercado Livre OAuth + Publishing.
Credentials are per-user (each empresa has its own ML app). Env vars are fallback defaults.
"""
import os
import uuid
import logging
import requests
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

ML_AUTH_BASE = "https://auth.mercadolivre.com.br"
ML_API_BASE = "https://api.mercadolibre.com"


def get_creds(user: dict | None) -> tuple[str, str, str]:
    """Return (client_id, client_secret, redirect_uri) preferring per-user config, falling back to env."""
    if user:
        cid = user.get("ml_client_id") or os.environ.get("ML_CLIENT_ID", "")
        sec = user.get("ml_client_secret") or os.environ.get("ML_CLIENT_SECRET", "")
        redir = user.get("ml_redirect_uri") or os.environ.get("ML_REDIRECT_URI", "")
    else:
        cid = os.environ.get("ML_CLIENT_ID", "")
        sec = os.environ.get("ML_CLIENT_SECRET", "")
        redir = os.environ.get("ML_REDIRECT_URI", "")
    return cid, sec, redir


def default_redirect_uri() -> str:
    """The Redirect URI the user should register in their ML app dashboard."""
    base = (os.environ.get("PUBLIC_BACKEND_URL") or "").rstrip("/")
    if not base:
        # Fallback: read frontend .env value if available
        base = ""
    return f"{base}/api/ml/callback" if base else "/api/ml/callback"


def is_configured(user: dict | None) -> bool:
    cid, sec, redir = get_creds(user)
    return bool(cid and sec and redir)


def build_authorize_url(user: dict, state: str) -> str:
    cid, _, redir = get_creds(user)
    return (
        f"{ML_AUTH_BASE}/authorization?response_type=code&client_id={cid}"
        f"&redirect_uri={redir}&state={state}"
    )


def exchange_code(user: dict, code: str) -> dict:
    cid, sec, redir = get_creds(user)
    resp = requests.post(
        f"{ML_API_BASE}/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": cid,
            "client_secret": sec,
            "code": code,
            "redirect_uri": redir,
        },
        headers={"accept": "application/json", "content-type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def refresh_token(user: dict, refresh: str) -> dict:
    cid, sec, _ = get_creds(user)
    resp = requests.post(
        f"{ML_API_BASE}/oauth/token",
        data={
            "grant_type": "refresh_token",
            "client_id": cid,
            "client_secret": sec,
            "refresh_token": refresh,
        },
        headers={"accept": "application/json", "content-type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def upload_picture_by_url(access_token: str, image_url: str) -> str | None:
    resp = requests.post(
        f"{ML_API_BASE}/pictures/items/upload",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"source": image_url},
        timeout=60,
    )
    if resp.ok:
        return resp.json().get("id")
    return None


def publish_item(access_token: str, item: dict) -> dict:
    resp = requests.post(
        f"{ML_API_BASE}/items",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json=item,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def get_user_id(access_token: str) -> str:
    """Retorna o user_id do vendedor autenticado (dono do token)."""
    resp = requests.get(
        f"{ML_API_BASE}/users/me",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return str(resp.json()["id"])


def list_item_ids(access_token: str, user_id: str, offset: int = 0, limit: int = 50) -> dict:
    """Lista os IDs dos anúncios (itens) do vendedor autenticado."""
    resp = requests.get(
        f"{ML_API_BASE}/users/{user_id}/items/search",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"offset": offset, "limit": limit},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()  # { results: [item_id, ...], paging: {...} }


def get_items_details(access_token: str, item_ids: list[str]) -> list[dict]:
    """Busca os detalhes (título, preço, estoque, status) de uma lista de item_ids.
    A API do ML aceita até 20 ids por chamada no endpoint multiget."""
    items: list[dict] = []
    for i in range(0, len(item_ids), 20):
        chunk = item_ids[i:i + 20]
        resp = requests.get(
            f"{ML_API_BASE}/items",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"ids": ",".join(chunk)},
            timeout=30,
        )
        resp.raise_for_status()
        for entry in resp.json():
            if entry.get("code") == 200 and entry.get("body"):
                items.append(entry["body"])
    return items


def fetch_seller_listings(access_token: str, limit: int = 50) -> list[dict]:
    """Função de alto nível: token -> lista de anúncios reais do vendedor,
    já no formato usado pelo painel do MercadoAuto."""
    user_id = get_user_id(access_token)
    search = list_item_ids(access_token, user_id, limit=limit)
    ids = search.get("results", [])
    raw_items = get_items_details(access_token, ids)

    listings = []
    for it in raw_items:
        listings.append({
            "sku": it.get("seller_custom_field") or it.get("id"),
            "ml_item_id": it.get("id"),
            "title": it.get("title"),
            "price": it.get("price"),
            "available_quantity": it.get("available_quantity"),
            "status": "published" if it.get("status") == "active" else
                       ("draft" if it.get("status") == "paused" else "error"),
            "permalink": it.get("permalink"),
            "thumbnail": it.get("thumbnail"),
        })
    return listings


def search_public_listings(query: str, limit: int = 10, site_id: str = "MLB") -> list[dict]:
    """Busca pública no catálogo do Mercado Livre (não exige token de usuário).
    Usada para sugerir título/preço a partir de anúncios reais parecidos."""
    resp = requests.get(
        f"{ML_API_BASE}/sites/{site_id}/search",
        params={"q": query, "limit": limit},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get("results", []):
        results.append({
            "title": item.get("title"),
            "price": item.get("price"),
            "permalink": item.get("permalink"),
            "thumbnail": item.get("thumbnail"),
        })
    return results


def suggest_from_sku(sku: str, brand: str = "") -> dict:
    """Gera uma sugestão de título e faixa de preço a partir de anúncios reais
    e já publicados no Mercado Livre que combinem com o SKU/marca informados."""
    query = " ".join(filter(None, [brand, sku])).strip()
    if not query:
        return {"found": False}
    try:
        results = search_public_listings(query, limit=10)
    except Exception as e:
        logger.error(f"Busca pública no ML falhou: {e}")
        results = []
    if not results:
        return {"found": False}
    prices = [r["price"] for r in results if r.get("price")]
    return {
        "found": True,
        "suggested_title": results[0]["title"],
        "price_min": min(prices) if prices else None,
        "price_max": max(prices) if prices else None,
        "sample_count": len(results),
    }


def mock_publish(product: dict) -> dict:
    fake_id = f"MLB{uuid.uuid4().hex[:10].upper()}"
    return {
        "id": fake_id,
        "permalink": f"https://produto.mercadolivre.com.br/{fake_id}",
        "status": "active",
        "mock": True,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }

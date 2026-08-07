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
    if not resp.ok:
        # A exceção padrão do requests não inclui o corpo da resposta —
        # que é onde o Mercado Livre explica o motivo real da rejeição
        # (ex: categoria inválida, falta de fotos, atributo obrigatório
        # ausente). Sem isso, o erro mostrado ao usuário é inútil.
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise ValueError(f"Mercado Livre recusou o anúncio ({resp.status_code}): {detail}")
    return resp.json()


def get_category_tree(category_id: str | None = None, site_id: str = "MLB", access_token: str | None = None) -> dict:
    """Lista as subcategorias de uma categoria (ou as categorias raiz, se
    category_id não for informado) — pra empresa navegar manualmente até a
    categoria exata do produto, igual ao fluxo de anúncio manual do próprio
    Mercado Livre. Mais confiável que a previsão automática por texto, que
    pode escorregar pra uma categoria completamente errada (como já vimos
    na prática).

    `access_token`: passa o token da conta conectada quando disponível —
    o endpoint de listar categorias raiz vem recusando chamadas anônimas
    com 403 (diferente de outros endpoints "públicos" que seguem
    funcionando sem token)."""
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    if not category_id:
        resp = requests.get(f"{ML_API_BASE}/sites/{site_id}/categories", headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return {
            "id": None,
            "name": None,
            "path": [],
            "children": [{"id": c.get("id"), "name": c.get("name")} for c in data],
            "is_leaf": False,
        }
    resp = requests.get(f"{ML_API_BASE}/categories/{category_id}", headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    children = data.get("children_categories") or []
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "path": [{"id": p.get("id"), "name": p.get("name")} for p in (data.get("path_from_root") or [])],
        "children": [{"id": c.get("id"), "name": c.get("name")} for c in children],
        "is_leaf": len(children) == 0,
    }


def predict_category(title: str, site_id: str = "MLB") -> str | None:
    """Descobre a categoria real do Mercado Livre a partir do título do
    produto, usando a API pública de sugestão de categoria — em vez de usar
    texto livre (ex: 'GERAL', vindo de uma planilha de ERP) como se fosse
    um category_id de verdade, o que o Mercado Livre sempre rejeitaria."""
    try:
        resp = requests.get(
            f"{ML_API_BASE}/sites/{site_id}/domain_discovery/search",
            params={"q": title[:100], "limit": 1},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if data and isinstance(data, list) and data[0].get("category_id"):
            return data[0]["category_id"]
    except Exception as e:
        logger.error(f"Previsão de categoria falhou para '{title}': {e}")
    return None


def get_category_attributes(category_id: str) -> list[dict]:
    """Lista os atributos da categoria (incluindo quais são obrigatórios),
    direto da API pública do Mercado Livre."""
    resp = requests.get(f"{ML_API_BASE}/categories/{category_id}/attributes", timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_category_name(category_id: str) -> str | None:
    """Nome legível da categoria (ex: 'Juntas e Vedações Automotivas') —
    usado só para exibir na tela, pra empresa perceber rápido se a
    previsão de categoria escorregou para algo claramente errado."""
    try:
        resp = requests.get(f"{ML_API_BASE}/categories/{category_id}", timeout=15)
        resp.raise_for_status()
        return resp.json().get("name")
    except Exception as e:
        logger.error(f"Falha ao buscar nome da categoria {category_id}: {e}")
        return None


def build_required_attributes(category_id: str, title: str, brand: str) -> list[dict]:
    """Monta, com melhor esforço, os atributos obrigatórios da categoria
    prevista — cada categoria do Mercado Livre pode exigir atributos
    diferentes (ex: 'family_name', 'BRAND'), e sem eles a publicação é
    recusada. Só preenche automaticamente quando o campo é texto livre
    (sem lista fechada de opções) — atributos com opções pré-definidas
    (ex: cor, voltagem) não são adivinhados, para não inventar dado
    técnico que não temos de verdade."""
    try:
        attrs = get_category_attributes(category_id)
    except Exception as e:
        logger.error(f"Falha ao buscar atributos da categoria {category_id}: {e}")
        return []

    result = []
    for attr in attrs:
        tags = attr.get("tags", {})
        attr_id = attr.get("id", "")
        is_known_hidden_required = attr_id.upper() in ("FAMILY_NAME",)
        if not (tags.get("required") or tags.get("catalog_required") or is_known_hidden_required):
            continue
        has_closed_values = bool(attr.get("values"))
        if attr_id == "BRAND":
            result.append({"id": attr_id, "value_name": brand or "Genérica"})
        elif attr_id in ("FAMILY_NAME", "MODEL", "LINE") and not has_closed_values:
            result.append({"id": attr_id, "value_name": (brand or title)[:60]})
        elif attr.get("value_type") == "string" and not has_closed_values:
            result.append({"id": attr_id, "value_name": "Não especificado"})
        # Atributos com lista fechada de valores (cor, voltagem, etc.) ficam
        # de fora — preencher errado é pior do que deixar o Mercado Livre
        # apontar exatamente qual falta, com o erro real já visível no site.
    return result


def update_item(access_token: str, ml_item_id: str, changes: dict) -> dict:
    """Atualiza um anúncio já publicado no Mercado Livre (PUT /items/{id}).
    `changes` deve conter só os campos que mudaram (ex: price, available_quantity,
    title). A descrição usa endpoint próprio — veja `update_item_description`."""
    resp = requests.put(
        f"{ML_API_BASE}/items/{ml_item_id}",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json=changes,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def update_item_description(access_token: str, ml_item_id: str, plain_text: str) -> None:
    """A descrição de um anúncio no Mercado Livre é atualizada por um
    endpoint separado do restante do item."""
    requests.put(
        f"{ML_API_BASE}/items/{ml_item_id}/description",
        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
        json={"plain_text": plain_text},
        timeout=60,
    ).raise_for_status()


def get_user_id(access_token: str) -> str:
    """Retorna o user_id do vendedor autenticado (dono do token)."""
    return str(get_user_info(access_token)["id"])


def get_user_info(access_token: str) -> dict:
    """Dados completos do vendedor autenticado (id, nickname, site_id...) —
    usado tanto para pegar o user_id quanto para confirmar, em tempo real,
    se a conexão com o Mercado Livre está realmente ativa."""
    resp = requests.get(
        f"{ML_API_BASE}/users/me",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


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


def search_catalog_product(query: str, site_id: str = "MLB", access_token: str | None = None) -> dict | None:
    """Busca no CATÁLOGO real de produtos do Mercado Livre — equivalente à
    aba "Por código" do anúncio manual (usa código/MPN pra achar o produto
    já cadastrado no catálogo deles, com categoria e atributos prontos,
    incluindo 'família'). Quando existe correspondência, publicar contra
    esse catalog_product_id evita ter que montar/adivinhar atributos —
    o Mercado Livre já sabe tudo sobre o produto."""
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    try:
        resp = requests.get(
            f"{ML_API_BASE}/products/search",
            params={"q": query, "site_id": site_id, "status": "active"},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        # O parâmetro 'status=active' da busca não garante, na prática, que
        # o resultado realmente esteja ativo (já vimos o Mercado Livre
        # recusar publicação com "not_active" mesmo assim) — confere de
        # verdade no código e pula qualquer resultado inativo, em vez de
        # confiar cegamente no primeiro item devolvido.
        for item in results:
            if item.get("status") == "active":
                return item
        return None
    except Exception as e:
        logger.error(f"Busca no catálogo falhou para '{query}': {e}")
        return None


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


def suggest_from_sku(title: str = "", sku: str = "", brand: str = "") -> dict:
    """Sugere título de referência e faixa de preço buscando anúncios reais
    e parecidos no catálogo público do Mercado Livre. Usa o TÍTULO do
    produto como termo principal — é o dado com chance real de bater com
    algo (nome/modelo do veículo, tipo de peça), diferente do SKU (código
    interno, sem relação com o texto de anúncios reais) ou da marca sozinha
    (frequentemente vazia). Só cai para marca+SKU como último recurso, caso
    a busca pelo título não encontre nada."""
    attempts = []
    clean_title = (title or "").strip()
    if clean_title:
        # Limita o tamanho da query — títulos muito longos (comuns em
        # autopeças, com vários veículos compatíveis) tendem a piorar a
        # busca em vez de ajudar; as primeiras palavras já carregam a
        # marca/modelo principal.
        attempts.append(clean_title[:100])
    fallback_query = " ".join(filter(None, [brand, sku])).strip()
    if fallback_query:
        attempts.append(fallback_query)

    for query in attempts:
        try:
            results = search_public_listings(query, limit=10)
        except Exception as e:
            logger.error(f"Busca pública no ML falhou para '{query}': {e}")
            results = []
        if results:
            prices = [r["price"] for r in results if r.get("price")]
            return {
                "found": True,
                "suggested_title": results[0]["title"],
                "price_min": min(prices) if prices else None,
                "price_max": max(prices) if prices else None,
                "sample_count": len(results),
                "query_used": query,
            }
    return {"found": False}


def mock_publish(product: dict) -> dict:
    fake_id = f"MLB{uuid.uuid4().hex[:10].upper()}"
    return {
        "id": fake_id,
        "permalink": f"https://produto.mercadolivre.com.br/{fake_id}",
        "status": "active",
        "mock": True,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }

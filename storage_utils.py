"""Object storage for uploaded images, stored directly in MongoDB.

Antes esse módulo dependia de um serviço externo específico da plataforma
Emergent (integrations.emergentagent.com), que só funciona dentro daquele
ambiente e estava falhando em produção (a variável EMERGENT_LLM_KEY nem
está configurada no Railway — por isso todo upload de foto de produto
falhava com erro 500, silenciosamente, sem nenhuma imagem chegar a ser
salva). Trocamos por uma coleção Mongo dedicada, que reaproveita a mesma
conexão que o resto do app já usa (MONGO_URL / DB_NAME) — nenhuma
credencial nova é necessária.

Guardamos o binário do arquivo direto no documento (campo `data`). Isso é
adequado para fotos de produto (poucos MB cada, e o limite de documento do
MongoDB é 16MB). Se o volume de imagens crescer muito, migrar para GridFS
ou S3 (boto3 já está nas dependências) é o próximo passo natural — mas essa
troca resolve o bug de produção agora, sem exigir nenhuma conta nova.
"""
import os
import logging
from datetime import datetime, timezone
from bson import Binary
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

logger = logging.getLogger(__name__)

APP_NAME = os.environ.get("APP_NAME", "mercadoauto")

_collection: AsyncIOMotorCollection | None = None


def init_storage() -> AsyncIOMotorCollection:
    """Inicializa (uma vez só) a coleção usada como storage. Não faz nenhuma
    chamada de rede além da conexão Mongo já usada pelo restante do app."""
    global _collection
    if _collection is not None:
        return _collection
    mongo_url = os.environ["MONGO_URL"]
    db_name = os.environ["DB_NAME"]
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]
    _collection = db["object_storage"]
    logger.info("Object storage (MongoDB collection) initialized")
    return _collection


async def put_object(path: str, data: bytes, content_type: str) -> dict:
    collection = init_storage()
    await collection.replace_one(
        {"path": path},
        {
            "path": path,
            "data": Binary(data),
            "content_type": content_type,
            "size": len(data),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        upsert=True,
    )
    return {"path": path, "size": len(data)}


async def get_object(path: str) -> tuple[bytes, str]:
    collection = init_storage()
    doc = await collection.find_one({"path": path})
    if not doc:
        raise FileNotFoundError(f"Arquivo não encontrado no storage: {path}")
    return bytes(doc["data"]), doc.get("content_type", "application/octet-stream")


MIME_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}

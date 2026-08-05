"""Armazenamento de arquivos (fotos de produto) via Cloudflare R2 —
compatível com a API do S3, usando boto3 (já presente no requirements.txt).
Substitui a versão anterior, que dependia de um serviço exclusivo da
plataforma Emergent e não funcionava fora dela.

Variáveis de ambiente necessárias (configure no Railway):
  R2_ACCOUNT_ID          — encontrado no painel do Cloudflare (R2 → Overview)
  R2_ACCESS_KEY_ID       — criado em R2 → Manage API Tokens
  R2_SECRET_ACCESS_KEY   — gerado junto com o Access Key ID (só aparece uma vez)
  R2_BUCKET_NAME         — nome do bucket criado no R2 (ex: "mercadoauto")

As funções públicas (init_storage, put_object, get_object) são `async def`
porque o server.py as chama com `await` — o boto3 em si é uma biblioteca
síncrona (bloqueante), então cada chamada roda numa thread separada via
`asyncio.to_thread`, para não travar o loop de eventos do FastAPI durante
um upload/download de arquivo."""
import os
import asyncio
import logging
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

APP_NAME = os.environ.get("APP_NAME", "mercadoauto")

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "mercadoauto")

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY):
        raise RuntimeError(
            "Credenciais do Cloudflare R2 não configuradas — defina "
            "R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY e "
            "R2_BUCKET_NAME nas variáveis de ambiente do Railway."
        )
    endpoint = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    _client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    return _client


def _put_object_sync(path: str, data: bytes, content_type: str) -> dict:
    client = _get_client()
    client.put_object(Bucket=R2_BUCKET_NAME, Key=path, Body=data, ContentType=content_type)
    return {"path": path, "size": len(data)}


def _get_object_sync(path: str) -> tuple[bytes, str]:
    client = _get_client()
    try:
        resp = client.get_object(Bucket=R2_BUCKET_NAME, Key=path)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            raise FileNotFoundError(path)
        raise
    data = resp["Body"].read()
    content_type = resp.get("ContentType", "application/octet-stream")
    return data, content_type


async def init_storage():
    """Mantido por compatibilidade com o startup do server.py (chamado com
    `await`) — com R2/S3 não existe uma etapa real de "inicialização"; aqui
    só confirmamos que as credenciais estão presentes, para o aviso no log
    ser claro se não estiverem."""
    await asyncio.to_thread(_get_client)
    logger.info("Cloudflare R2 storage pronto (bucket: %s)", R2_BUCKET_NAME)
    return True


async def put_object(path: str, data: bytes, content_type: str) -> dict:
    return await asyncio.to_thread(_put_object_sync, path, data, content_type)


async def get_object(path: str) -> tuple[bytes, str]:
    return await asyncio.to_thread(_get_object_sync, path)


MIME_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


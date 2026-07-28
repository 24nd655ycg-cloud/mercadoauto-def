"""AI content generation with Claude, via a real Anthropic API key (console.anthropic.com).
Substitui a versão anterior, que dependia do pacote interno `emergentintegrations`
e só funcionava dentro da plataforma Emergent."""
import os
import json
import logging
import re
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic | None:
    global _client
    if not ANTHROPIC_API_KEY:
        return None
    if _client is None:
        _client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _client


SYSTEM_PROMPT = (
    "Você é um especialista em copywriting de anúncios de e-commerce do Mercado Livre Brasil. "
    "Sua tarefa é gerar títulos e descrições otimizadas para SEO e conversão a partir dos "
    "dados brutos do produto. Sempre em português brasileiro, tom direto e profissional. "
    "O título deve ter no máximo 60 caracteres, começar com a marca ou categoria e conter "
    "palavras-chave de busca. A descrição deve ser estruturada em blocos curtos com bullet "
    "points, destacando características técnicas, dimensões, garantia e diferenciais. "
    "SEMPRE responda estritamente com um objeto JSON válido no formato: "
    '{"title": "...", "description": "..."} sem markdown, sem texto adicional.'
)


async def generate_listing_content(raw_title: str, raw_description: str, brand: str = "", category: str = "") -> dict:
    """Generates optimized title & description. Returns dict with keys `title` and `description`.
    Faz fallback silencioso para os dados brutos se a chave não estiver configurada
    ou se a chamada à API falhar — o anúncio nunca fica bloqueado por causa da IA."""
    client = _get_client()
    if not client:
        return {"title": raw_title, "description": raw_description}

    prompt = (
        f"Produto:\nTítulo bruto: {raw_title}\nMarca: {brand or 'não informada'}\n"
        f"Categoria: {category or 'não informada'}\nDescrição bruta: {raw_description}\n\n"
        "Gere título e descrição otimizados."
    )

    try:
        resp = await client.messages.create(
            model=MODEL_NAME,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            return {
                "title": (data.get("title") or raw_title)[:60],
                "description": data.get("description") or raw_description,
            }
    except Exception as e:
        logger.error(f"AI generation failed: {e}")

    return {"title": raw_title, "description": raw_description}

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

# Usado quando a empresa ainda não configurou o próprio modelo de descrição
# (aba Configurações → Descrição modelo).
DEFAULT_DESCRIPTION_TEMPLATE = """{titulo_produto}

✔ APLICAÇÃO
{aplicacao}

✔ MARCA / FABRICANTE
{marca}

✔ CÓDIGO DE REFERÊNCIA
{sku} {codigo_referencia_ml}

✔ CONDIÇÃO
Peça nova, original ou equivalente de mercado, pronta para uso.

✔ DIFERENCIAIS
{diferenciais}

✔ GARANTIA
{garantia}

—
Dúvidas? Chame antes de comprar — respondemos rápido."""

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


TEMPLATE_SYSTEM_PROMPT = (
    "Você preenche um MODELO DE DESCRIÇÃO de anúncio de autopeças para o Mercado Livre Brasil, "
    "e também escreve um título curto e otimizado para busca (máximo 60 caracteres). "
    "A empresa forneceu o modelo de descrição exatamente como deve ser seguido — respeite a "
    "estrutura, as quebras de linha, os títulos de bloco (ex: ------ CARACTERÍSTICAS ------) e "
    "qualquer texto fixo do modelo (incluindo rodapé/assinatura da empresa) EXATAMENTE como está "
    "escrito. Troque apenas os placeholders entre colchetes ou chaves pelas informações reais "
    "fornecidas.\n\n"
    "REGRAS OBRIGATÓRIAS:\n"
    "1. NUNCA invente informação técnica que não foi fornecida (marca, código, ano, motorização, "
    "compatibilidade). Se um dado não estiver disponível nas informações fornecidas, escreva "
    "literalmente 'Não informado' nesse campo — não tente adivinhar.\n"
    "2. Só use dados de aplicação/veículo (montadora, modelo, motorização, anos) se eles vierem "
    "explicitamente do título do produto ou da referência real encontrada no Mercado Livre.\n"
    "3. Anos no formato 'XX>' (ex: '12>') significam 'a partir de XX', sem ano final — escreva "
    "como '(20XX - atual)'.\n"
    "4. Não remova nem reescreva blocos do modelo — apenas preencha os placeholders.\n"
    "5. Responda estritamente com um objeto JSON válido no formato "
    '{"title": "...", "description": "..."} — sem markdown, sem texto adicional, sem crases.'
)


async def generate_listing_from_template(
    template: str,
    raw_title: str,
    sku: str,
    brand: str = "",
    category: str = "",
    ml_reference_title: str | None = None,
) -> dict:
    """Preenche o template de descrição (da empresa, ou o padrão) e gera o
    título otimizado — em UMA única chamada à IA, para evitar duas idas
    sequenciais que podem estourar timeout. Nunca inventa dado técnico —
    usa 'Não informado' quando não sabe."""
    client = _get_client()
    if not client:
        return {"title": raw_title[:60], "description": template.replace("{titulo_produto}", raw_title)}

    reference_note = (
        f'Referência real encontrada no Mercado Livre (anúncio parecido já publicado): "{ml_reference_title}".'
        if ml_reference_title else
        "Nenhuma referência real foi encontrada no Mercado Livre para este produto."
    )

    prompt = (
        f"MODELO DE DESCRIÇÃO A SEGUIR (preencha exatamente esta estrutura):\n---\n{template}\n---\n\n"
        f"DADOS REAIS DISPONÍVEIS:\n"
        f"Título/nome do produto: {raw_title}\n"
        f"SKU (código interno): {sku or 'não informado'}\n"
        f"Marca: {brand or 'não informada'}\n"
        f"Categoria: {category or 'não informada'}\n"
        f"{reference_note}\n\n"
        "Preencha o modelo de descrição com esses dados (use 'Não informado' onde não houver "
        "dado real) e gere também um título curto otimizado para busca."
    )

    try:
        resp = await client.messages.create(
            model=MODEL_NAME,
            max_tokens=1200,
            system=TEMPLATE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if data.get("description"):
                return {
                    "title": (data.get("title") or raw_title)[:60],
                    "description": data["description"],
                }
    except Exception as e:
        logger.error(f"Template AI generation failed: {e}")

    return {"title": raw_title[:60], "description": template.replace("{titulo_produto}", raw_title)}

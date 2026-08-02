"""AI content generation — suporta Anthropic (Claude) ou OpenAI, escolhido
pela variável de ambiente AI_PROVIDER ("anthropic" por padrão, ou "openai").
Configure a chave correspondente (ANTHROPIC_API_KEY ou OPENAI_API_KEY)."""
import os
import json
import logging
import re

logger = logging.getLogger(__name__)

AI_PROVIDER = os.environ.get("AI_PROVIDER", "anthropic").strip().lower()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

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


_anthropic_client = None
_openai_client = None


def _get_anthropic_client():
    global _anthropic_client
    if not ANTHROPIC_API_KEY:
        return None
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic
        _anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if not OPENAI_API_KEY:
        return None
    if _openai_client is None:
        from openai import AsyncOpenAI
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def ai_configured() -> bool:
    """True se o provedor escolhido em AI_PROVIDER tem uma chave configurada."""
    if AI_PROVIDER == "openai":
        return bool(OPENAI_API_KEY)
    return bool(ANTHROPIC_API_KEY)


async def _call_ai(system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> str | None:
    """Chama o provedor de IA configurado (Anthropic ou OpenAI) e devolve o
    texto bruto da resposta, ou None se não houver chave configurada ou a
    chamada falhar (o chamador decide o fallback nesses casos)."""
    try:
        if AI_PROVIDER == "openai":
            client = _get_openai_client()
            if not client:
                return None
            resp = await client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.choices[0].message.content

        client = _get_anthropic_client()
        if not client:
            return None
        resp = await client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")
    except Exception as e:
        logger.error(f"Chamada à IA ({AI_PROVIDER}) falhou: {e}")
        return None


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


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
    """Generates optimized title & description. Returns dict com `title` e `description`.
    Faz fallback silencioso para os dados brutos se nenhuma chave estiver configurada
    ou se a chamada à API falhar — o anúncio nunca fica bloqueado por causa da IA."""
    if not ai_configured():
        return {"title": raw_title, "description": raw_description}

    prompt = (
        f"Produto:\nTítulo bruto: {raw_title}\nMarca: {brand or 'não informada'}\n"
        f"Categoria: {category or 'não informada'}\nDescrição bruta: {raw_description}\n\n"
        "Gere título e descrição otimizados."
    )
    text = await _call_ai(SYSTEM_PROMPT, prompt, max_tokens=1000)
    data = _extract_json(text)
    if data:
        return {
            "title": (data.get("title") or raw_title)[:60],
            "description": data.get("description") or raw_description,
        }
    return {"title": raw_title, "description": raw_description}


TEMPLATE_SYSTEM_PROMPT = (
    "Você preenche um MODELO DE DESCRIÇÃO de anúncio de autopeças para o Mercado Livre Brasil, "
    "e também escreve um título curto e otimizado para busca (máximo 60 caracteres). "
    "A empresa forneceu o modelo de descrição exatamente como deve ser seguido — respeite a "
    "estrutura, as quebras de linha, os títulos de bloco (ex: ------ CARACTERÍSTICAS ------) e "
    "qualquer texto fixo do modelo (incluindo rodapé/assinatura da empresa) EXATAMENTE como está "
    "escrito.\n\n"
    "REGRAS OBRIGATÓRIAS:\n"
    "1. TODO placeholder entre colchetes [ASSIM] ou chaves {assim} DEVE ser substituído — "
    "pela informação real, se você tiver, ou pela palavra 'Não informado', se não tiver. "
    "PROIBIDO devolver a descrição final com qualquer colchete [ ] ou chave { } ainda visível — "
    "isso é considerado uma falha grave.\n"
    "2. NUNCA invente informação técnica que não foi fornecida (marca, código, ano, motorização, "
    "compatibilidade). Se um dado não estiver disponível nas informações fornecidas, escreva "
    "literalmente 'Não informado' nesse campo específico — não tente adivinhar, mas também não "
    "deixe o colchete original ali.\n"
    "3. Só use dados de aplicação/veículo (montadora, modelo, motorização, anos) se eles vierem "
    "explicitamente do título do produto ou da referência real encontrada no Mercado Livre.\n"
    "4. Anos no formato 'XX>' (ex: '12>') significam 'a partir de XX', sem ano final — escreva "
    "como '(20XX - atual)'.\n"
    "5. Não remova nem reescreva o texto fixo do modelo — apenas preencha os placeholders.\n\n"
    "EXEMPLO — se o modelo tem a linha:\n"
    "Marca: [Fabricante] (Original [Montadora])\n"
    "e você só sabe a montadora (Fiat) mas não o fabricante da peça específica, escreva:\n"
    "Marca: Não informado (Original Fiat)\n"
    "— nunca deixe '[Fabricante]' ou '[Montadora]' no texto final.\n\n"
    "Responda estritamente com um objeto JSON válido no formato "
    '{"title": "...", "description": "..."} — sem markdown, sem texto adicional, sem crases.'
)


_PLACEHOLDER_PATTERN = re.compile(r"\[[A-ZÀ-Ú][^\]\n]{0,60}\]|\{[a-z_]{2,40}\}")


def _has_unfilled_placeholders(text: str) -> bool:
    return bool(_PLACEHOLDER_PATTERN.search(text or ""))


_PLACEHOLDER_PATTERN = re.compile(r"\[[^\]\n]{1,60}\]|\{[^}\n]{1,60}\}")


def _has_unfilled_placeholders(text: str) -> bool:
    return bool(_PLACEHOLDER_PATTERN.search(text or ""))


def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _fill_remaining_placeholders(text: str, raw_title: str, sku: str, brand: str) -> str:
    """Rede de segurança determinística: substitui qualquer [placeholder] ou
    {placeholder} que a IA tenha deixado sem preencher. Usa dado real do
    produto quando o nome do placeholder indica claramente qual campo é
    (ex: contém 'sku'/'codigo' -> usa o SKU real); caso contrário, usa
    'Não informado' — nunca deixa um colchete chegar até o usuário."""

    def _replace(match: re.Match) -> str:
        inner = _strip_accents(match.group(0)[1:-1].strip().lower())
        if "sku" in inner or "codigo" in inner or "referencia" in inner:
            return sku if sku else "Não informado"
        if "marca" in inner or "fabricante" in inner or "montadora" in inner:
            return brand if brand else "Não informado"
        if "nome" in inner or "produto" in inner or "peca" in inner or "titulo" in inner:
            return raw_title if raw_title else "Não informado"
        return "Não informado"

    return _PLACEHOLDER_PATTERN.sub(_replace, text or "")


async def generate_listing_from_template(
    template: str,
    raw_title: str,
    sku: str,
    brand: str = "",
    category: str = "",
    ml_reference_title: str | None = None,
) -> dict:
    """Preenche o template de descrição (da empresa, ou o padrão) e gera o
    título otimizado — em UMA única chamada à IA. A IA cuida do conteúdo
    (aplicação, diferenciais); o CÓDIGO garante, depois, que nenhum
    placeholder fica sem preencher — nunca inventa dado técnico, usa
    'Não informado' quando não sabe."""
    if not ai_configured():
        return {"title": raw_title[:60], "description": template.replace("{titulo_produto}", raw_title), "template_filled": False}

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

    text = await _call_ai(TEMPLATE_SYSTEM_PROMPT, prompt, max_tokens=1200)
    data = _extract_json(text)

    if data and data.get("description"):
        description = data["description"]
        had_gaps = _has_unfilled_placeholders(description)
        if had_gaps:
            # O código fecha qualquer lacuna que a IA tenha deixado — nunca
            # chega colchete/chave até o usuário, independente do que a IA
            # tenha "esquecido".
            description = _fill_remaining_placeholders(description, raw_title, sku, brand)
        return {
            "title": (data.get("title") or raw_title)[:60],
            "description": description,
            "template_filled": True,
            "ai_completed_fully": not had_gaps,
        }

    # A IA não respondeu nada aproveitável: preenche o template inteiro
    # usando só a rede de segurança (garante que nada fica em branco/colchete,
    # mesmo sem nenhuma contribuição da IA).
    return {
        "title": raw_title[:60],
        "description": _fill_remaining_placeholders(template.replace("{titulo_produto}", raw_title), raw_title, sku, brand),
        "template_filled": True,
        "ai_completed_fully": False,
    }

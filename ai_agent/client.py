"""Small Groq client with a local fallback for the web UI.

The external LLM is only an interpretation layer. If dependencies or API keys
are missing, the site must keep working and return a useful local explanation.
"""

from __future__ import annotations

import os
from typing import List


def _load_env_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _local_fallback(text: str) -> str:
    t = text.lower()
    if "nmi" in t or "ari" in t:
        return (
            "NMI показывает, насколько найденные кластеры согласуются с истинными метками "
            "по информации о разбиении. ARI сравнивает пары объектов и строже штрафует "
            "случайные совпадения. В этой системе gt используется только для оценки качества, "
            "а не для построения консенсус кластеризации."
        )
    if "sdgca" in t:
        return (
            "SDGCA строит взвешенную матрицу совместной встречаемости и дополнительно учитывает "
            "сходство и различие между объектами. Модифицированная версия меняет схему весов и "
            "графовую диффузию. Сравнивать их нужно по NMI, ARI, F-score и runtime на одинаковых "
            "датасетах."
        )
    if "m" in t or "параметр" in t:
        return (
            "m задаёт, сколько базовых кластеризаций используется для построения консенсуса. "
            "Малое m работает быстрее, но может быть менее устойчивым. Большое m обычно стабильнее, "
            "но дороже по времени. Подбирать m лучше на validation-наборе, а test использовать "
            "только для финальной проверки."
        )
    return (
        "Внешний Groq API сейчас недоступен, поэтому сработал локальный режим. "
        "Система может объяснять диагностику датасета, параметры запуска и метрики, но не должна "
        "подбирать параметры по test split. Для честного эксперимента используйте validation/test "
        "разделение по датасетам."
    )


def _get_groq_client():
    """Return (client, model) or raise ImportError/ValueError if unavailable."""
    _load_env_if_available()
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    from groq import Groq  # may raise ImportError
    return Groq(api_key=api_key), model


def ask_llm(prompt: str) -> str:
    """Send a single user prompt to the LLM."""
    try:
        client, model = _get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except (ValueError, ImportError):
        return _local_fallback(prompt)
    except Exception as exc:
        return f"{_local_fallback(prompt)}\n\nGroq API error: {exc}"


def ask_llm_with_messages(messages: List[dict]) -> str:
    """Send a full messages array (system + history + user) to the LLM.

    Each element must be a dict with 'role' and 'content' keys.
    Falls back to local heuristics if the API is unavailable.
    """
    try:
        client, model = _get_groq_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except (ValueError, ImportError):
        # Extract last user message for the fallback heuristic
        user_text = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        return _local_fallback(user_text)
    except Exception as exc:
        user_text = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        return f"{_local_fallback(user_text)}\n\nGroq API error: {exc}"

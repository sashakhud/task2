"""
Суммаризация текста через OpenRouter API.

Модуль инкапсулирует взаимодействие с LLM.
Для смены провайдера достаточно реализовать интерфейс BaseSummarizer.

Расширение:
  - Другие провайдеры (OpenAI, Anthropic, local LLM) — создайте
    новый класс, наследующий BaseSummarizer
  - Другие промпты — передайте prompt_template в конструктор
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ── Промпты ─────────────────────────────────────────────────────────

LEGAL_PROMPT_TEMPLATE = """Ты — опытный юридический ассистент. Тебе дан текст, извлечённый из документа. Текст мог быть получен через OCR с фотографии, поэтому в нём могут встречаться опечатки, пропуски символов и артефакты распознавания — это нормально, работай с тем, что есть.

Задача: сделай структурированную суммаризацию на русском языке, извлекая максимум полезной информации.

Правила:
1. Опиши основную тему и предмет регулирования документа (1-2 предложения).
2. Если в тексте есть статьи, пункты или части — кратко опиши суть каждой найденной.
3. Сохраняй точную юридическую терминологию.
4. Если упоминаются ссылки на нормативные акты (федеральные законы, постановления, указы и т.д.) — перечисли их с номерами и датами.
5. Длина суммаризации должна быть пропорциональна объёму текста:
   — короткий текст (1 страница или фрагмент): 30-100 слов;
   — средний текст (2-5 страниц): 100-300 слов;
   — длинный текст (более 5 страниц): 300-600 слов.
6. НЕ отказывайся обрабатывать текст. Даже если текст неполный, содержит OCR-артефакты или выглядит как фрагмент — всё равно извлеки и опиши всю доступную информацию.
7. Отвечай ТОЛЬКО текстом суммаризации. Не пиши фразы вроде «текст нечитаемый», «невозможно обработать», «данный текст не является документом».
8. Используй только стандартные символы русского и латинского алфавитов, цифры и знаки препинания. Не используй символы-заменители, эмодзи или спецсимволы вроде ▯ или �.

Текст:
{text}

Суммаризация:"""

GENERAL_PROMPT_TEMPLATE = """Тебе дан текст, извлечённый из документа (возможно, через OCR — могут быть артефакты распознавания).
Сделай краткую суммаризацию на русском языке. Извлеки основные тезисы и ключевую информацию.
Даже если текст короткий или неполный — опиши всё, что можно извлечь.
Длина суммаризации — пропорционально объёму текста: от 30 слов для короткого фрагмента до 500 слов для большого документа.
Не используй символы-заменители (�), эмодзи или спецсимволы. Только стандартный текст.

Текст:
{text}

Суммаризация:"""


# ── Базовый класс ───────────────────────────────────────────────────

class BaseSummarizer(ABC):
    """Интерфейс суммаризатора — для подмены провайдера."""

    @abstractmethod
    def summarize(self, text: str) -> str:
        ...


# ── OpenRouter ──────────────────────────────────────────────────────

class OpenRouterSummarizer(BaseSummarizer):
    """Суммаризация через OpenRouter API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        model: str = "stepfun/step-3.5-flash:free",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        document_type: str = "legal",
        prompt_template: Optional[str] = None,
        max_text_chars: int = 30000,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_text_chars = max_text_chars

        if prompt_template:
            self._prompt_template = prompt_template
        elif document_type == "legal":
            self._prompt_template = LEGAL_PROMPT_TEMPLATE
        else:
            self._prompt_template = GENERAL_PROMPT_TEMPLATE

    def summarize(self, text: str) -> str:
        """
        Суммаризировать текст через LLM.

        Если текст слишком длинный, он обрезается до max_text_chars.
        При ошибке API возвращается сообщение об ошибке.
        """
        if not text or not text.strip():
            return "[Текст пуст — суммаризация невозможна]"

        # Обрезка длинного текста
        if len(text) > self.max_text_chars:
            text = text[: self.max_text_chars] + "\n\n[…текст обрезан…]"

        prompt = self._prompt_template.format(text=text)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/document-summarizer",
        }

        # Повторные попытки при rate-limit
        for attempt in range(3):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    # Убираем символы-заменители (�) и прочий мусор
                    content = content.replace("\ufffd", "").replace("▯", "")
                    return content

                if response.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "Rate-limit (429), ожидание %d сек…", wait,
                    )
                    time.sleep(wait)
                    continue

                logger.error(
                    "API вернул %d: %s", response.status_code, response.text,
                )
                return f"[Ошибка API: {response.status_code}]"

            except requests.RequestException as e:
                logger.error("Сетевая ошибка: %s", e)
                if attempt < 2:
                    time.sleep(2)
                    continue
                return f"[Сетевая ошибка: {e}]"

        return "[Не удалось получить ответ от API после нескольких попыток]"


"""
Конфигурация проекта суммаризации юридических документов.

Все настройки собраны в одном месте для удобного управления.
Для расширения: добавьте новые параметры в соответствующий раздел.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    """Центральная конфигурация приложения."""

    # ── OpenRouter API ──────────────────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    # Рекомендуемая модель для русского языка:
    #   stepfun/step-3.5-flash:free — лучший баланс качества и скорости
    model_name: str = "stepfun/step-3.5-flash:free"
    max_tokens: int = 2048
    temperature: float = 0.3

    # ── Пути ────────────────────────────────────────────────────────
    documents_folder: str = "./documents"
    output_file: str = "./summary_output.txt"
    output_json: str = "./summary_output.json"
    # Папка для сохранения извлечённых OCR-текстов (для отладки и проверки)
    extracted_texts_folder: str = "./extracted_texts"

    # ── Поддерживаемые расширения → тип обработчика ─────────────────
    #    Чтобы добавить новый формат, достаточно:
    #    1) Создать класс-обработчик, наследующий BaseProcessor
    #    2) Добавить расширение сюда
    #    3) Зарегистрировать обработчик в ProcessorRegistry (main.py)
    extension_to_type: Dict[str, str] = field(default_factory=lambda: {
        ".pdf": "pdf",
        ".jpg": "image",
        ".jpeg": "image",
        ".webp": "image",
        ".png": "image",
        # Будущие форматы:
        # ".docx": "docx",
        # ".html": "html",
        # ".rtf":  "rtf",
    })

    # ── OCR ─────────────────────────────────────────────────────────
    # Движок OCR: "paddleocr" (рекомендуется), "pytesseract" или "easyocr"
    #   paddleocr   — нейросетевой, лучший для фото книг/документов
    #   pytesseract — классический, только для чистых сканов
    #   easyocr     — нейросетевой, слабее paddle на плотном тексте
    ocr_engine: str = "paddleocr"
    ocr_languages: List[str] = field(default_factory=lambda: ["rus", "eng"])
    # Для easyocr используются коды ["ru", "en"]
    easyocr_languages: List[str] = field(default_factory=lambda: ["ru", "en"])

    # ── Предобработка изображений для OCR ───────────────────────────
    # Для paddleocr предобработка не нужна (свой внутренний пайплайн)
    # Для tesseract: мягкая предобработка без агрессивной бинаризации
    image_preprocess: bool = True      # включить/выключить предобработку
    image_deskew: bool = True          # автоматическое выравнивание
    image_binarize: bool = False       # бинаризация — ВЫКЛЮЧЕНА для фото!
    image_denoise: bool = True         # удаление шума

    # ── Формат вывода ───────────────────────────────────────────────
    # "text", "json", "markdown"  — легко добавлять новые
    output_format: str = "text"

    # ── Суммаризация ────────────────────────────────────────────────
    # Максимальное количество символов текста, отправляемого в LLM
    max_text_chars: int = 30000
    # Тип документа для формирования промпта
    document_type: str = "legal"

    def __post_init__(self):
        """Загружаем API-ключ из окружения, если не задан явно."""
        if not self.openrouter_api_key:
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")


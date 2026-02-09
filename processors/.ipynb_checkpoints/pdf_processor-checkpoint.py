"""
Обработчик PDF-документов.

Поддерживает:
  - Текстовые PDF (извлечение текста напрямую)
  - Сканированные PDF (fallback на OCR через ImageProcessor)
"""

from __future__ import annotations

import logging
from typing import List

from .base import BaseProcessor, ProcessedDocument

logger = logging.getLogger(__name__)


class PDFProcessor(BaseProcessor):
    """Извлечение текста из PDF-файлов с помощью PyMuPDF (fitz)."""

    def __init__(self, ocr_fallback: BaseProcessor | None = None):
        """
        Args:
            ocr_fallback: обработчик изображений для сканированных PDF.
                          Если None — сканированные страницы пропускаются.
        """
        self._ocr_fallback = ocr_fallback

    @staticmethod
    def supported_extensions() -> List[str]:
        return [".pdf"]

    def extract(self, file_path: str) -> ProcessedDocument:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF не установлен: pip install PyMuPDF")
            return ProcessedDocument(filename=file_path, text="")

        pages_text: List[str] = []

        try:
            doc = fitz.open(file_path)
            page_count = len(doc)

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")

                # Если текст пустой — возможно, это скан
                if not text.strip() and self._ocr_fallback is not None:
                    logger.info(
                        "Страница %d в %s без текста → OCR",
                        page_num, file_path,
                    )
                    # Рендерим страницу в изображение и прогоняем OCR
                    pix = page.get_pixmap(dpi=300)
                    img_path = f"_tmp_page_{page_num}.png"
                    pix.save(img_path)
                    result = self._ocr_fallback.extract(img_path)
                    text = result.text
                    # Удаляем временный файл
                    import os
                    try:
                        os.remove(img_path)
                    except OSError:
                        pass

                pages_text.append(text)

            doc.close()

            full_text = "\n\n".join(pages_text)
            return ProcessedDocument(
                filename=file_path,
                text=full_text,
                page_count=page_count,
                metadata={"source": "PyMuPDF"},
            )

        except Exception as e:
            logger.error("Ошибка обработки PDF %s: %s", file_path, e)
            return ProcessedDocument(filename=file_path, text="")


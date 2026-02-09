#!/usr/bin/env python3
"""
Суммаризатор юридических документов.

Точка входа — обрабатывает папку с документами (PDF, JPG, WEBP и др.),
извлекает текст, находит ссылки на нормативные акты и генерирует
суммаризацию с помощью LLM через OpenRouter.

Использование:
    python main.py                               # интерактивный ввод пути
    python main.py --folder ./docs               # указать папку
    python main.py --folder ./docs --format json  # формат вывода

Переменные окружения:
    OPENROUTER_API_KEY — API-ключ OpenRouter (обязательно)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from config import Config
from processors import ImageProcessor, PDFProcessor
from processors.base import ProcessorRegistry
from extractors import LegalReferenceExtractor
from summarizer import OpenRouterSummarizer
from output import OutputFormatter

# ── Логирование ─────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("summarizer")


# ── Пайплайн обработки ─────────────────────────────────────────────

class DocumentPipeline:
    """
    Оркестратор: сканирует папку, обрабатывает каждый файл,
    извлекает ссылки, суммаризирует и форматирует результат.
    """

    def __init__(self, config: Config):
        self.config = config

        # Создаём обработчики
        self.image_processor = ImageProcessor(
            engine=config.ocr_engine,
            languages=(
                config.easyocr_languages
                if config.ocr_engine == "easyocr"
                else config.ocr_languages
            ),
            preprocess=config.image_preprocess,
            binarize=config.image_binarize,
            denoise=config.image_denoise,
        )
        self.pdf_processor = PDFProcessor(ocr_fallback=self.image_processor)

        # Регистрируем обработчики
        self.registry = ProcessorRegistry()
        self.registry.register(self.pdf_processor)
        self.registry.register(self.image_processor)

        # Экстрактор ссылок
        self.ref_extractor = LegalReferenceExtractor()

        # Суммаризатор
        self.summarizer = OpenRouterSummarizer(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            model=config.model_name,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            document_type=config.document_type,
            max_text_chars=config.max_text_chars,
        )

        # Форматирование
        self.formatter = OutputFormatter(output_format=config.output_format)

    # ── Основной метод ──────────────────────────────────────────────

    def run(self, folder: str) -> str:
        """
        Обработать все документы в папке и вернуть отформатированный результат.
        """
        folder_path = Path(folder)
        if not folder_path.is_dir():
            logger.error("Папка не найдена: %s", folder_path)
            return ""

        files = self._scan_folder(folder_path)
        if not files:
            logger.warning("Поддерживаемые файлы не найдены в %s", folder_path)
            return ""

        logger.info("Найдено файлов: %d", len(files))

        results: List[Dict[str, Any]] = []
        for idx, file_path in enumerate(files, start=1):
            logger.info(
                "[%d/%d] Обработка: %s", idx, len(files), file_path.name,
            )
            result = self._process_file(file_path)
            results.append(result)

        # Форматирование
        formatted = self.formatter.format_results(results)

        # Сохранение
        ext_map = {"text": ".txt", "json": ".json", "markdown": ".md"}
        ext = ext_map.get(self.config.output_format, ".txt")
        output_path = str(
            Path(self.config.output_file).with_suffix(ext)
        )
        self.formatter.save(formatted, output_path)

        return formatted

    # ── Вспомогательные ─────────────────────────────────────────────

    def _scan_folder(self, folder: Path) -> List[Path]:
        """Собрать поддерживаемые файлы из папки (без рекурсии)."""
        supported = set(self.config.extension_to_type.keys())
        files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in supported
        )
        return files

    def _process_file(self, file_path: Path) -> Dict[str, Any]:
        """Обработать один файл: извлечь текст → сохранить → ссылки → суммаризация."""
        ext = file_path.suffix.lower()
        processor = self.registry.get_processor(ext)

        if processor is None:
            return {
                "filename": file_path.name,
                "error": f"Нет обработчика для расширения {ext}",
            }

        # 1. Извлечение текста
        doc = processor.extract(str(file_path))

        # 2. Сохранение извлечённого текста в отдельную папку
        self._save_extracted_text(file_path, doc.text)

        if doc.is_empty:
            return {
                "filename": file_path.name,
                "error": "Не удалось извлечь текст из документа",
            }

        logger.info(
            "  Текст извлечён: %d символов, %d стр.",
            len(doc.text), doc.page_count,
        )

        # 3. Извлечение ссылок на НПА
        refs = self.ref_extractor.extract(doc.text)
        if not refs.is_empty:
            logger.info("  Найдено ссылок на НПА: %d", len(refs.references))

        # 4. Суммаризация через LLM
        logger.info("  Отправка на суммаризацию…")
        summary = self.summarizer.summarize(doc.text)

        return {
            "filename": file_path.name,
            "summary": summary,
            "references_lines": refs.summary_lines(),
            "references_dict": refs.to_dict(),
            "text_length": len(doc.text),
            "page_count": doc.page_count,
            "error": None,
        }

    def _save_extracted_text(self, source_file: Path, text: str) -> None:
        """
        Сохранить извлечённый текст в папку extracted_texts/.
        Имя файла = имя исходного файла + .txt
        Сохраняется ВСЕГДА (даже если текст пустой — чтобы было видно проблему).
        """
        try:
            out_dir = Path(self.config.extracted_texts_folder)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Имя: оригинальное имя + .txt  (например "photo1.webp.txt")
            out_path = out_dir / (source_file.name + ".txt")

            if text and text.strip():
                content = (
                    f"# Извлечённый текст из: {source_file.name}\n"
                    f"# Символов: {len(text)}\n"
                    f"{'─' * 60}\n\n"
                    f"{text}\n"
                )
            else:
                content = (
                    f"# Извлечённый текст из: {source_file.name}\n"
                    f"# ВНИМАНИЕ: текст пуст — OCR не смог извлечь данные.\n"
                    f"# Проверьте качество исходного изображения.\n"
                )

            out_path.write_text(content, encoding="utf-8")
            logger.info("  Текст сохранён: %s", out_path)

        except Exception as e:
            logger.warning("  Не удалось сохранить текст: %s", e)


# ── CLI ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Суммаризация юридических документов в папке",
    )
    parser.add_argument(
        "--folder", "-f",
        type=str,
        default=None,
        help="Путь к папке с документами",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Путь к файлу с результатом",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "markdown"],
        default="text",
        help="Формат вывода (по умолчанию: text)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Модель OpenRouter (по умолчанию: stepfun/step-3.5-flash:free)",
    )
    parser.add_argument(
        "--ocr-engine",
        type=str,
        choices=["paddleocr", "pytesseract", "easyocr"],
        default="paddleocr",
        help="Движок OCR (по умолчанию: paddleocr — лучший для фото документов)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API-ключ OpenRouter (или переменная OPENROUTER_API_KEY)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    # Применяем аргументы CLI
    if args.api_key:
        config.openrouter_api_key = args.api_key
    if args.model:
        config.model_name = args.model
    if args.output:
        config.output_file = args.output
    if args.format:
        config.output_format = args.format
    if args.ocr_engine:
        config.ocr_engine = args.ocr_engine

    # Путь к папке
    folder = args.folder
    if not folder:
        folder = input("Введите путь к папке с документами: ").strip()
        if not folder:
            print("Ошибка: путь к папке не указан.")
            sys.exit(1)
    config.documents_folder = folder

    # Проверяем API-ключ
    if not config.openrouter_api_key:
        print(
            "Ошибка: укажите API-ключ OpenRouter.\n"
            "  Вариант 1: set OPENROUTER_API_KEY=ваш_ключ\n"
            "  Вариант 2: python main.py --api-key ваш_ключ"
        )
        sys.exit(1)

    # Запуск
    pipeline = DocumentPipeline(config)
    result = pipeline.run(folder)

    if result:
        print(result)
    else:
        print("Не удалось обработать документы.")
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
Обработчик изображений (JPG, WEBP, PNG) с OCR.

Поддерживает три движка (в порядке рекомендации):
  1. paddleocr  — РЕКОМЕНДУЕМЫЙ для фотографий документов.
                  Нейросетевой OCR, отлично справляется с кривизной
                  страниц, неравномерным освещением, русским текстом.
  2. pytesseract — классический OCR. Хорош для чистых сканов,
                   плох для фотографий книг/документов.
  3. easyocr     — нейросетевой, но слабее PaddleOCR на плотном тексте.

Предобработка:
  - Для PaddleOCR предобработка ОТКЛЮЧЕНА (у него своя внутренняя).
  - Для Tesseract: мягкая предобработка (без агрессивной бинаризации).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from .base import BaseProcessor, ProcessedDocument

logger = logging.getLogger(__name__)


class ImageProcessor(BaseProcessor):
    """OCR-обработчик изображений."""

    def __init__(
        self,
        engine: str = "paddleocr",
        languages: Optional[List[str]] = None,
        preprocess: bool = True,
        binarize: bool = False,
        denoise: bool = True,
    ):
        """
        Args:
            engine:     "paddleocr" (рекомендуется), "pytesseract" или "easyocr"
            languages:  список языков
                        paddleocr:   не требуется (задаётся через lang="ru")
                        pytesseract: ["rus", "eng"]
                        easyocr:     ["ru", "en"]
            preprocess: предобработка изображения (для paddleocr игнорируется)
            binarize:   бинаризация — по умолчанию ВЫКЛЮЧЕНА
                        (для фотографий она вредит!)
            denoise:    лёгкое удаление шума
        """
        self.engine = engine
        self.preprocess = preprocess
        self.binarize = binarize
        self.denoise = denoise

        # Языки по движкам
        if engine == "easyocr":
            self.languages = languages or ["ru", "en"]
        elif engine == "pytesseract":
            self.languages = languages or ["rus", "eng"]
        else:
            # paddleocr — язык задаётся отдельно
            self.languages = languages or ["ru"]

        # Ленивая инициализация тяжёлых объектов
        self._paddleocr_engine = None
        self._easyocr_reader = None

    @staticmethod
    def supported_extensions() -> List[str]:
        return [".jpg", ".jpeg", ".webp", ".png"]

    # ── Основной метод ──────────────────────────────────────────────

    def extract(self, file_path: str) -> ProcessedDocument:
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            logger.error("Не удалось открыть изображение %s: %s", file_path, e)
            return ProcessedDocument(filename=file_path, text="")

        # Увеличиваем разрешение, если изображение маленькое
        image = self._upscale_if_needed(image)

        # Предобработка — ТОЛЬКО для tesseract, и мягкая
        if self.engine == "pytesseract" and self.preprocess:
            image = self._preprocess_for_tesseract(image)

        # OCR
        if self.engine == "paddleocr":
            text = self._ocr_paddleocr(file_path, image)
        elif self.engine == "easyocr":
            text = self._ocr_easyocr(image)
        else:
            text = self._ocr_pytesseract(image)

        return ProcessedDocument(
            filename=file_path,
            text=text,
            metadata={"ocr_engine": self.engine},
        )

    # ── Предобработка ───────────────────────────────────────────────

    @staticmethod
    def _upscale_if_needed(image: Image.Image, min_width: int = 1500) -> Image.Image:
        """
        Если изображение слишком маленькое, увеличить в 2×.
        Маленькие изображения — главная причина плохого OCR.
        """
        w, h = image.size
        if w < min_width:
            scale = max(2, min_width // w + 1)
            new_size = (w * scale, h * scale)
            image = image.resize(new_size, Image.LANCZOS)
            logger.info("  Изображение увеличено: %dx%d → %dx%d", w, h, *new_size)
        return image

    def _preprocess_for_tesseract(self, image: Image.Image) -> Image.Image:
        """
        Мягкая предобработка для Tesseract.
        НЕ делаем агрессивную бинаризацию — Tesseract делает свою внутреннюю.
        Только: контраст + резкость + опционально лёгкий денойз.
        """
        # Перевод в grayscale
        gray = image.convert("L")

        # Повышение контраста
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(1.5)

        # Повышение резкости
        gray = gray.filter(ImageFilter.SHARPEN)

        # Лёгкое удаление шума (медианный фильтр 3x3 — минимальный)
        if self.denoise:
            gray = gray.filter(ImageFilter.MedianFilter(size=3))

        # Бинаризация — ТОЛЬКО если явно включена (по умолчанию выключена)
        if self.binarize:
            try:
                import cv2
                img_array = np.array(gray)
                # Метод Оцу — автоматический порог, гораздо лучше
                # адаптивного для книжных страниц
                _, binary = cv2.threshold(
                    img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                gray = Image.fromarray(binary)
            except ImportError:
                logger.warning("opencv-python не установлен, бинаризация пропущена")

        return gray

    # ── PaddleOCR ───────────────────────────────────────────────────

    def _ocr_paddleocr(self, file_path: str, image: Image.Image) -> str:
        """
        PaddleOCR v3+ — рекомендуемый движок.
        Нейросетевой OCR с отличным распознаванием русского текста.
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            logger.error(
                "paddleocr не установлен.\n"
                "  Установка: pip install paddlepaddle==3.2.0 paddleocr==3.3.0\n"
                "  Подробнее: https://github.com/PaddlePaddle/PaddleOCR"
            )
            return ""

        try:
            if self._paddleocr_engine is None:
                self._paddleocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang="ru",
                )

            # PaddleOCR v3+ лучше работает с путём к файлу,
            # но для upscaled-изображений нужен numpy array.
            # Пробуем оба варианта.
            results = None

            # Вариант 1: передаём путь к файлу (оригинал без upscale)
            # Вариант 2: передаём numpy array (с upscale)
            img_input = np.array(image)

            try:
                results = self._paddleocr_engine.ocr(img_input)
            except TypeError:
                # Некоторые версии не принимают numpy — передаём путь
                logger.info("  PaddleOCR: fallback на путь к файлу")
                results = self._paddleocr_engine.ocr(file_path)

            if not results:
                logger.warning("  PaddleOCR вернул пустой результат")
                return ""

            # Диагностика формата (для отладки)
            logger.debug("  PaddleOCR raw type: %s", type(results))
            if results:
                logger.debug("  PaddleOCR results[0] type: %s", type(results[0]))
                if isinstance(results, list) and len(results) > 0:
                    first = results[0]
                    if isinstance(first, dict):
                        logger.debug("  PaddleOCR dict keys: %s", list(first.keys()))
                    elif isinstance(first, list) and len(first) > 0:
                        logger.debug("  PaddleOCR first item type: %s", type(first[0]))

            return self._parse_paddle_results(results)

        except Exception as e:
            logger.error("Ошибка PaddleOCR: %s", e)
            return ""

    def _parse_paddle_results(self, results) -> str:
        """
        Универсальный парсер результатов PaddleOCR.
        Поддерживает все известные форматы вывода (v2, v3, v3.3+).
        """
        lines = []

        if not results:
            return ""

        # ── Определяем формат и извлекаем текст ────────────────────

        for page_or_item in results:

            # ── Формат v3.3+ (page-level dict) ─────────────────────
            # Один dict на страницу:
            # {
            #   'input_path': ...,
            #   'rec_text': ['строка1', 'строка2', ...],
            #   'rec_score': [0.98, 0.95, ...],
            #   'dt_polys': [[[x,y],...], ...],
            # }
            if isinstance(page_or_item, dict):
                texts = page_or_item.get("rec_text") or page_or_item.get("rec_texts", [])
                scores = page_or_item.get("rec_score") or page_or_item.get("rec_scores", [])
                polys = page_or_item.get("dt_polys", [])

                # rec_text — список строк
                if isinstance(texts, list) and texts:
                    if not isinstance(scores, list):
                        scores = [1.0] * len(texts)
                    for i, text in enumerate(texts):
                        text = str(text).strip()
                        conf = float(scores[i]) if i < len(scores) else 1.0
                        if text and conf > 0.3:
                            # Вычисляем Y-позицию для сортировки
                            y = 0
                            if polys and i < len(polys):
                                try:
                                    poly = polys[i]
                                    y = float(np.mean([p[1] for p in poly]))
                                except (IndexError, TypeError):
                                    y = i  # порядок как есть
                            else:
                                y = i
                            lines.append((y, text))
                    continue

                # rec_text — единичная строка
                if isinstance(texts, str) and texts.strip():
                    lines.append((0, texts.strip()))
                    continue

            # ── Формат v2 / v3 (list of lines per page) ────────────
            # results = [ [ [bbox, (text, conf)], [bbox, (text, conf)], ... ] ]
            if isinstance(page_or_item, list):
                for line_item in page_or_item:
                    try:
                        if isinstance(line_item, dict):
                            # dict внутри списка страницы
                            t = line_item.get("rec_text", "") or line_item.get("text", "")
                            c = line_item.get("rec_score", 1.0) or line_item.get("score", 1.0)
                            if t and float(c) > 0.3:
                                lines.append((0, str(t).strip()))

                        elif isinstance(line_item, (list, tuple)) and len(line_item) == 2:
                            bbox = line_item[0]
                            text_data = line_item[1]

                            if isinstance(text_data, (list, tuple)) and len(text_data) >= 2:
                                text = str(text_data[0]).strip()
                                conf = float(text_data[1])
                            elif isinstance(text_data, str):
                                text = text_data.strip()
                                conf = 1.0
                            else:
                                continue

                            if text and conf > 0.3:
                                y = (bbox[0][1] + bbox[2][1]) / 2 if bbox else 0
                                lines.append((y, text))

                    except (IndexError, TypeError, ValueError) as e:
                        logger.debug("Пропуск строки OCR: %s", e)
                        continue

        if not lines:
            logger.warning("  PaddleOCR: парсер не извлёк ни одной строки")
            # Последняя попытка — извлечь текст напрямую через str()
            raw = str(results)
            if len(raw) > 100:
                logger.debug("  Raw results (первые 500 символов): %s", raw[:500])
            return ""

        # Сортируем по вертикальной позиции (сверху вниз)
        lines.sort(key=lambda x: x[0])
        return "\n".join(text for _, text in lines).strip()

    # ── Pytesseract ─────────────────────────────────────────────────

    def _ocr_pytesseract(self, image: Image.Image) -> str:
        try:
            import pytesseract
        except ImportError:
            logger.error("pytesseract не установлен: pip install pytesseract")
            return ""

        try:
            lang_str = "+".join(self.languages)  # "rus+eng"
            # Улучшенные параметры:
            #   --psm 6: предполагаем единый блок текста
            #   --oem 1: только LSTM (нейросетевой движок, точнее legacy)
            config = "--psm 6 --oem 1"
            text = pytesseract.image_to_string(image, lang=lang_str, config=config)
            return text.strip()
        except Exception as e:
            logger.error("Ошибка pytesseract: %s", e)
            return ""

    # ── EasyOCR ─────────────────────────────────────────────────────

    def _ocr_easyocr(self, image: Image.Image) -> str:
        try:
            import easyocr
        except ImportError:
            logger.error("easyocr не установлен: pip install easyocr")
            return ""

        try:
            if self._easyocr_reader is None:
                self._easyocr_reader = easyocr.Reader(
                    self.languages, gpu=False,
                )
            img_array = np.array(image)
            results = self._easyocr_reader.readtext(img_array, detail=0)
            return "\n".join(results).strip()
        except Exception as e:
            logger.error("Ошибка easyocr: %s", e)
            return ""

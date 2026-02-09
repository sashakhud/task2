"""
Извлечение ссылок на нормативные правовые акты из юридического текста.

Поддерживаемые типы:
  - Федеральные законы (ФЗ)
  - Постановления Конституционного Суда РФ (КС РФ)
  - Постановления Правительства РФ
  - Указы Президента РФ
  - Распоряжения Правительства РФ
  - Законы РФ / РСФСР

Расширение:
  Чтобы добавить новый тип НПА, добавьте элемент в PATTERNS
  и обработку в _extract_by_pattern().
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LegalReference:
    """Одна найденная ссылка на НПА."""
    ref_type: str          # "Федеральный закон", "Постановление КС РФ", …
    number: str            # "41-ФЗ", "12-П", …
    date: str              # "20 марта 2011"  или  "20.03.2011"
    raw_match: str = ""    # исходный фрагмент текста


@dataclass
class ExtractionResult:
    """Результат извлечения всех ссылок из документа."""
    references: List[LegalReference] = field(default_factory=list)

    def to_dict(self) -> Dict:
        grouped: Dict[str, List[Dict]] = {}
        for ref in self.references:
            grouped.setdefault(ref.ref_type, []).append({
                "number": ref.number,
                "date": ref.date,
                "raw": ref.raw_match,
            })
        return grouped

    @property
    def is_empty(self) -> bool:
        return len(self.references) == 0

    def summary_lines(self) -> List[str]:
        """Человеко-читаемый список найденных ссылок."""
        lines: List[str] = []
        grouped = self.to_dict()
        for ref_type, items in grouped.items():
            lines.append(f"  {ref_type}:")
            for item in items:
                lines.append(f"    — № {item['number']} от {item['date']}")
        return lines


class LegalReferenceExtractor:
    """
    Извлекает ссылки на нормативные правовые акты (НПА) из текста.

    Используются регулярные выражения с поддержкой нескольких
    форматов дат и написания.
    """

    # Месяцы в родительном падеже → число (для нормализации)
    MONTHS_MAP = {
        "января": "01", "февраля": "02", "марта": "03",
        "апреля": "04", "мая": "05", "июня": "06",
        "июля": "07", "августа": "08", "сентября": "09",
        "октября": "10", "ноября": "11", "декабря": "12",
    }

    # Общий фрагмент: дата в текстовом формате
    _DATE_TEXT = r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})\s*г(?:ода)?\."

    # Общий фрагмент: дата в числовом формате
    _DATE_NUM = r"(\d{1,2})\.(\d{2})\.(\d{4})"

    # ── Паттерны для каждого типа НПА ──────────────────────────────

    PATTERNS: Dict[str, List[re.Pattern]] = {}

    @classmethod
    def _compile_patterns(cls):
        """Компилируем паттерны один раз."""
        if cls.PATTERNS:
            return

        # --- Федеральные законы ---
        cls.PATTERNS["Федеральный закон"] = [
            # «Федерального закона от 20 марта 2011 г. № 41-ФЗ»
            re.compile(
                r"[Фф]едеральн\w{0,4}\s+закон\w{0,3}\s+от\s+"
                + cls._DATE_TEXT
                + r"\s*№\s*([\d]+-ФЗ)",
                re.IGNORECASE,
            ),
            # «Федерального закона от 20.03.2011 № 41-ФЗ»
            re.compile(
                r"[Фф]едеральн\w{0,4}\s+закон\w{0,3}\s+от\s+"
                + cls._DATE_NUM
                + r"\s*№\s*([\d]+-ФЗ)",
                re.IGNORECASE,
            ),
            # Короткая форма: «от 20 марта 2011 г. № 41-ФЗ» (когда контекст «в ред.»)
            re.compile(
                r"в\s+ред(?:акции)?\.?\s+.*?от\s+"
                + cls._DATE_TEXT
                + r"\s*№\s*([\d]+-ФЗ)",
                re.IGNORECASE,
            ),
        ]

        # --- Постановления КС РФ ---
        cls.PATTERNS["Постановление КС РФ"] = [
            re.compile(
                r"[Пп]остановлени\w{0,3}\s+(?:Конституционного\s+Суда|КС)\s+(?:РФ|Российской\s+Федерации)\s+от\s+"
                + cls._DATE_TEXT
                + r"\s*№\s*([\w\d/.-]+)",
                re.IGNORECASE,
            ),
            re.compile(
                r"[Пп]остановлени\w{0,3}\s+(?:Конституционного\s+Суда|КС)\s+(?:РФ|Российской\s+Федерации)\s+от\s+"
                + cls._DATE_NUM
                + r"\s*№\s*([\w\d/.-]+)",
                re.IGNORECASE,
            ),
        ]

        # --- Постановления Правительства РФ ---
        cls.PATTERNS["Постановление Правительства РФ"] = [
            re.compile(
                r"[Пп]остановлени\w{0,3}\s+Правительства\s+(?:РФ|Российской\s+Федерации)\s+от\s+"
                + cls._DATE_TEXT
                + r"\s*№\s*([\w\d/.-]+)",
                re.IGNORECASE,
            ),
            re.compile(
                r"[Пп]остановлени\w{0,3}\s+Правительства\s+(?:РФ|Российской\s+Федерации)\s+от\s+"
                + cls._DATE_NUM
                + r"\s*№\s*([\w\d/.-]+)",
                re.IGNORECASE,
            ),
        ]

        # --- Указы Президента РФ ---
        cls.PATTERNS["Указ Президента РФ"] = [
            re.compile(
                r"[Уу]каз\w{0,3}\s+Президента\s+(?:РФ|Российской\s+Федерации)\s+от\s+"
                + cls._DATE_TEXT
                + r"\s*№\s*([\w\d/.-]+)",
                re.IGNORECASE,
            ),
            re.compile(
                r"[Уу]каз\w{0,3}\s+Президента\s+(?:РФ|Российской\s+Федерации)\s+от\s+"
                + cls._DATE_NUM
                + r"\s*№\s*([\w\d/.-]+)",
                re.IGNORECASE,
            ),
        ]

        # --- Распоряжения Правительства РФ ---
        cls.PATTERNS["Распоряжение Правительства РФ"] = [
            re.compile(
                r"[Рр]аспоряжени\w{0,3}\s+Правительства\s+(?:РФ|Российской\s+Федерации)\s+от\s+"
                + cls._DATE_TEXT
                + r"\s*№\s*([\w\d/.-]+)",
                re.IGNORECASE,
            ),
        ]

        # --- Определения КС РФ ---
        cls.PATTERNS["Определение КС РФ"] = [
            re.compile(
                r"[Оо]пределени\w{0,3}\s+(?:Конституционного\s+Суда|КС)\s+(?:РФ|Российской\s+Федерации)\s+от\s+"
                + cls._DATE_TEXT
                + r"\s*№\s*([\w\d/.-]+)",
                re.IGNORECASE,
            ),
        ]

    # ── Публичные методы ────────────────────────────────────────────

    def extract(self, text: str) -> ExtractionResult:
        """
        Извлечь все ссылки на НПА из текста.

        Returns:
            ExtractionResult с найденными ссылками
        """
        self._compile_patterns()
        result = ExtractionResult()
        seen = set()  # дедупликация

        for ref_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    groups = match.groups()
                    raw = match.group(0)

                    date_str, number = self._parse_groups(groups)
                    key = (ref_type, number, date_str)
                    if key in seen:
                        continue
                    seen.add(key)

                    result.references.append(LegalReference(
                        ref_type=ref_type,
                        number=number,
                        date=date_str,
                        raw_match=raw.strip(),
                    ))

        return result

    # ── Внутренние методы ───────────────────────────────────────────

    @classmethod
    def _parse_groups(cls, groups: tuple) -> tuple[str, str]:
        """
        Извлечь дату и номер из групп regex-матча.

        Формат групп зависит от паттерна:
          Текстовая дата: (день, месяц_слово, год, номер)
          Числовая дата:  (день, месяц_число, год, номер)
        """
        number = groups[-1]  # номер всегда последний

        if len(groups) == 4:
            # Текстовая дата или числовая дата
            day, month_or_num, year = groups[0], groups[1], groups[2]
            if month_or_num in cls.MONTHS_MAP:
                date_str = f"{day} {month_or_num} {year}"
            else:
                # числовой формат
                date_str = f"{day}.{month_or_num}.{year}"
        elif len(groups) >= 5:
            # Может быть паттерн с «в ред.» — берём первые 3 группы
            day, month_or_num, year = groups[-4], groups[-3], groups[-2]
            if month_or_num in cls.MONTHS_MAP:
                date_str = f"{day} {month_or_num} {year}"
            else:
                date_str = f"{day}.{month_or_num}.{year}"
        else:
            date_str = "дата не определена"

        return date_str, number


"""
Форматирование и сохранение результатов суммаризации.

Поддерживаемые форматы вывода:
  - text     — простой текстовый файл
  - json     — структурированный JSON
  - markdown — Markdown-документ

Расширение:
  Добавьте новый формат, реализовав метод _format_<name>()
  и зарегистрировав его в FORMATTERS.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


@staticmethod
def _make_separator(char: str = "═", length: int = 80) -> str:
    return char * length


class OutputFormatter:
    """Форматирует и сохраняет результаты."""

    def __init__(self, output_format: str = "text"):
        self.output_format = output_format.lower()

    # ── Публичный интерфейс ─────────────────────────────────────────

    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Отформатировать список результатов в строку."""
        formatter = {
            "text": self._format_text,
            "json": self._format_json,
            "markdown": self._format_markdown,
        }.get(self.output_format, self._format_text)
        return formatter(results)

    def save(self, content: str, file_path: str) -> None:
        """Сохранить отформатированный результат в файл."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info("Результат сохранён: %s", path.resolve())

    # ── Форматы ─────────────────────────────────────────────────────

    @staticmethod
    def _format_text(results: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        sep = "═" * 80

        lines.append(sep)
        lines.append("  СУММАРИЗАЦИЯ ЮРИДИЧЕСКИХ ДОКУМЕНТОВ")
        lines.append(sep)

        for i, res in enumerate(results, start=1):
            lines.append("")
            lines.append(f"  [{i}] {res['filename']}")
            lines.append("─" * 80)

            if res.get("error"):
                lines.append(f"  ⚠ Ошибка: {res['error']}")
            else:
                # Суммаризация
                lines.append("")
                lines.append("  СУММАРИЗАЦИЯ:")
                for line in res.get("summary", "").split("\n"):
                    lines.append(f"  {line}")

                # Ссылки на НПА
                refs = res.get("references_lines", [])
                if refs:
                    lines.append("")
                    lines.append("  ССЫЛКИ НА НОРМАТИВНЫЕ АКТЫ:")
                    for line in refs:
                        lines.append(f"  {line}")

            lines.append("")
            lines.append(sep)

        return "\n".join(lines)

    @staticmethod
    def _format_json(results: List[Dict[str, Any]]) -> str:
        # Убираем нефорсируемые в JSON поля
        clean = []
        for res in results:
            entry = {
                "filename": res["filename"],
                "summary": res.get("summary", ""),
                "references": res.get("references_dict", {}),
                "error": res.get("error"),
            }
            clean.append(entry)
        return json.dumps(clean, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_markdown(results: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        lines.append("# Суммаризация юридических документов\n")

        for i, res in enumerate(results, start=1):
            lines.append(f"## {i}. {res['filename']}\n")

            if res.get("error"):
                lines.append(f"> ⚠ Ошибка: {res['error']}\n")
            else:
                lines.append("### Суммаризация\n")
                lines.append(res.get("summary", "") + "\n")

                refs = res.get("references_lines", [])
                if refs:
                    lines.append("### Ссылки на нормативные акты\n")
                    for line in refs:
                        lines.append(line)
                    lines.append("")

            lines.append("---\n")

        return "\n".join(lines)


"""
Базовый класс для обработчиков документов.

Расширение:
    Для нового формата создайте класс, наследующий BaseProcessor,
    реализуйте extract() и supported_extensions(),
    затем зарегистрируйте его в ProcessorRegistry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Type


@dataclass
class ProcessedDocument:
    """Результат обработки одного документа."""
    filename: str
    text: str
    page_count: int = 1
    metadata: Dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text or not self.text.strip()


class BaseProcessor(ABC):
    """Абстрактный обработчик документа."""

    @abstractmethod
    def extract(self, file_path: str) -> ProcessedDocument:
        """Извлечь текст и метаданные из файла."""
        ...

    @staticmethod
    @abstractmethod
    def supported_extensions() -> List[str]:
        """Список поддерживаемых расширений (с точкой)."""
        ...


class ProcessorRegistry:
    """
    Реестр обработчиков.

    Позволяет регистрировать обработчики и получать нужный
    по расширению файла.

    Пример:
        registry = ProcessorRegistry()
        registry.register(PDFProcessor)
        registry.register(ImageProcessor)
        proc = registry.get_processor(".pdf")
    """

    def __init__(self):
        self._map: Dict[str, BaseProcessor] = {}

    def register(self, processor: BaseProcessor) -> None:
        """Зарегистрировать экземпляр обработчика для его расширений."""
        for ext in processor.supported_extensions():
            self._map[ext.lower()] = processor

    def get_processor(self, extension: str) -> BaseProcessor | None:
        """Вернуть обработчик для заданного расширения или None."""
        return self._map.get(extension.lower())

    @property
    def supported_extensions(self) -> List[str]:
        return list(self._map.keys())


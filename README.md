# Educational Plagiarism Detector

## Описание

Система для определения плагиата в студенческих работах, использующая различные методы сравнения текстов (cosine similarity, sequence matching, n-grams).

## Как установить

### Необходимые компоненты

- Python 3.8+
- pip or conda

### Установка

```bash
git clone <repo-url>
cd <project-name>
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Как использовать

### Базовый пример

```python
from src.main import analyze_data

result = analyze_data("data/sample.csv")
print(result)
```

### Advanced Usage

[Другие сложные примеры]

## Структура проекта

## Зависимости

- ... >= 1.20.0
- ... >= 1.3.0
- ... >= 1.0.0

## Тестирование

Run tests with:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Лицензия

**MIT License**

## Author

Никита Лецколюк

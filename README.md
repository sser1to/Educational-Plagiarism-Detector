# Educational Plagiarism Detector

## Описание

Система для определения плагиата в студенческих работах, использующая различные методы сравнения текстов (cosine similarity, sequence matching, n-grams).

### Основной функционал

- Загрузка студенческих работ (txt, pdf файлы)
- Предварительная обработка (tokenization, lemmatization)
- Сравнение текстов несколькими методами:
  - Cosine similarity с TF-IDF
  - Longest common subsequence
  - N-gram matching
- Вывод процента похожести между работами
- Визуализация результатов в виде матрицы похожести


## Как установить

### Необходимые компоненты

- Python 3.11
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

```
.
├── src/                 # Source code
│   ├── __init__.py
│   ├── main.py
│   └── utils.py
├── tests/              # Unit tests
│   ├── __init__.py
│   └── test_main.py
├── data/               # Data files
│   └── sample.csv
├── docs/               # Documentation
├── scripts/            # Utility scripts
├── .github/workflows/  # CI/CD
├── README.md
├── requirements.txt
└── .gitignore
```

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

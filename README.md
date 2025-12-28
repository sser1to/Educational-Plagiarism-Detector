# Educational Plagiarism Detector

## Описание

Система для определения плагиата в студенческих работах, использующая различные методы сравнения текстов (cosine similarity, sequence matching, n-grams).

### Основной функционал

- Загрузка студенческих работ (txt, pdf, docx)
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
- pip

### Установка

```bash
# Клонирование репозитория
git clone https://github.com/sser1to/Educational-Plagiarism-Detector.git

cd Educational-Plagiarism-Detector

# Создание виртуального окружения
python -m venv venv

# Активация виртуального окружения
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
```

### Быстрый старт

```bash
# Простой анализ примеров
python run.py data/

# Запуск интерактивных примеров
python scripts/example.py
```

## Как использовать

### Базовый пример

```python
from src.main import analyze_plagiarism

# Анализ всех файлов в директории
detector = analyze_plagiarism('uploads/', threshold=0.7)

# Или более детально
from src.main import PlagiarismDetector

detector = PlagiarismDetector(language='english')
detector.load_documents('uploads/')
detector.process_documents()
detector.compare_documents()
detector.print_report(threshold=0.7)
detector.visualize_results(save_dir='results/')
```

### Расширенное использование

```python
from src.main import PlagiarismDetector

# Создание детектора
detector = PlagiarismDetector(language='english')

# Загрузка конкретных файлов
detector.load_documents(['student1.txt', 'student2.txt', 'student3.txt'])

# Обработка с настройками
detector.process_documents(
    remove_stopwords=True,
    lemmatize=True
)

# Сравнение всех документов
results = detector.compare_documents()

# Детальное сравнение двух документов
details = detector.compare_two_documents('student1.txt', 'student2.txt')
print(f"Схожесть: {details['average_similarity']:.1%}")

# Генерация отчета
report = detector.generate_report(threshold=0.7)
for pair in report['suspicious_pairs']:
    print(f"{pair['document1']} ↔ {pair['document2']}: {pair['similarity']:.1%}")
```

### Использование из командной строки

```bash
# Базовый анализ
python -m src.main uploads/

# Запуск примеров
python scripts/example.py
```

## Структура проекта

```
.
├── src/                            # Исходный код
│   ├── __init__.py
│   ├── main.py                     # Главный модуль с PlagiarismDetector
│   ├── document_loader.py          # Загрузка документов (txt, pdf, docx)
│   ├── text_processor.py           # Обработка текста (токенизация, лемматизация)
│   ├── similarity_calculator.py    # Алгоритмы сравнения
│   └── visualizer.py               # Визуализация результатов
├── tests/                          # Юнит-тесты
│   ├── __init__.py
│   └── test_main.py                # Тесты всех компонентов
├── data/                           # Данные для примеров
├── docs/                           # Документация
│   └── API.md                      # Документация API
├── scripts/                        # Утилиты и примеры
│   └── example.py                  # Примеры использования
├── uploads/                        # Директория для загрузки документов
├── results/                        # Результаты анализа (графики)
├── README.md
├── requirements.txt                # Зависимости проекта
├── .gitignore
└── LICENSE
```

## Зависимости

- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- nltk >= 3.6.0
- pymorphy2 >= 0.9.1
- PyPDF2 >= 3.0.0
- python-docx >= 0.8.11
- pytest >= 7.0.0
- flake8>=6.0.0
- black>=23.0.0

## Тестирование

Запуск тестов:

```bash
# Запуск всех тестов
pytest

# Запуск с покрытием
pytest --cov=src tests/

# Запуск с подробным выводом
pytest -v

# Запуск конкретного теста
pytest tests/test_main.py::TestPlagiarismDetector::test_compare_documents
```

## Возможности

✅ **Поддержка форматов:** TXT, PDF, DOCX  
✅ **Многоязычность:** Английский и Русский  
✅ **Множественные метрики:** Cosine Similarity, LCS, N-grams, Sequence Matcher  
✅ **Визуализация:** Тепловые карты, графики, отчеты  
✅ **Автоматические отчеты:** Выявление подозрительных пар  
✅ **Гибкие настройки:** Пороги схожести, опции обработки текста  
✅ **Командная строка:** Быстрый анализ без написания кода  
✅ **Полное тестирование:** Unit-тесты для всех компонентов

## Примеры результатов

Система генерирует несколько типов визуализаций:

1. **Матрица схожести** - показывает попарную схожесть всех документов
2. **Отчет о плагиате** - выделяет подозрительные пары с высокой схожестью
3. **Сравнительные графики** - детальное сравнение различных метрик
4. **Распределение схожести** - гистограмма всех значений схожести

<img width="1440" height="840" alt="example1" src="https://github.com/user-attachments/assets/7ac8d52e-c9b3-434e-99f5-2bafac4ba7c2" />

<img width="1440" height="840" alt="example2" src="https://github.com/user-attachments/assets/f982b65d-53f8-4bec-9590-34ae48c46a74" />

## FAQ

**Q: Какой порог схожести использовать?**  
A: Для академических работ рекомендуется 0.6-0.7. Для строгого контроля - 0.5.

**Q: Поддерживается ли русский язык?**  
A: Да, установите pymorphy2 и используйте `language='russian'`.

**Q: Можно ли анализировать большое количество документов?**  
A: Да, но время обработки увеличится пропорционально количеству пар для сравнения.

**Q: Как интерпретировать результаты?**  
A: 0.9+ = высокий риск, 0.7-0.9 = средний риск, 0.5-0.7 = низкий риск, <0.5 = оригинально.

## Дополнительная документация

[API Documentation](docs/API.md) - Подробное описание всех классов и методов

[Examples](scripts/example.py) - Интерактивные примеры использования

## Лицензия

**MIT License**

## Author

Никита Лецколюк

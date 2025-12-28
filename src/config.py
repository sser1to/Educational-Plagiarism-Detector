"""
Конфигурация для Educational Plagiarism Detector.

Этот файл содержит настройки по умолчанию, которые можно изменять.
"""

# Настройки языка
DEFAULT_LANGUAGE = 'english'  # 'english' или 'russian'

# Настройки обработки текста
TEXT_PROCESSING = {
    'remove_stopwords': True,  # Удалять стоп-слова
    'lemmatize': True,         # Применять лемматизацию
    'remove_punctuation': True # Удалять пунктуацию
}

# Настройки детектирования плагиата
PLAGIARISM_DETECTION = {
    'threshold': 0.7,          # Порог для определения плагиата (0-1)
    'high_risk_threshold': 0.9,# Порог высокого риска
    'medium_risk_threshold': 0.7# Порог среднего риска
}

# Настройки визуализации
VISUALIZATION = {
    'figure_size': (12, 8),    # Размер графиков
    'colormap': 'RdYlGn',      # Цветовая схема
    'dpi': 300,                # Разрешение сохраненных графиков
    'show_annotations': True   # Показывать значения на графиках
}

# Настройки TF-IDF
TFIDF_CONFIG = {
    'min_df': 1,               # Минимальная частота документа
    'max_df': 1.0,             # Максимальная частота документа
    'ngram_range': (1, 2)      # Диапазон n-грамм (unigrams и bigrams)
}

# Настройки N-gram similarity
NGRAM_CONFIG = {
    'bigram_n': 2,             # Размер биграмм
    'trigram_n': 3             # Размер триграмм
}

# Пути
PATHS = {
    'uploads_dir': 'uploads',  # Директория для загрузки документов
    'results_dir': 'results',  # Директория для результатов
    'data_dir': 'data'         # Директория для данных
}

# Поддерживаемые форматы файлов
SUPPORTED_FORMATS = ['.txt', '.pdf', '.docx', '.doc']

# Настройки отчетов
REPORT_CONFIG = {
    'max_pairs_display': 10,   # Максимальное количество пар в консольном отчете
    'show_detailed_metrics': True  # Показывать детальные метрики
}

# Настройки логирования
LOGGING = {
    'level': 'INFO',           # Уровень логирования: DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

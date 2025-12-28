"""
Модуль для обработки текстов: токенизация, лемматизация, очистка.
"""
import re
import string
from typing import List, Set
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextProcessor:
    """Класс для предварительной обработки текстов."""
    
    def __init__(self, language: str = 'english'):
        """
        Инициализация процессора текста.
        
        Args:
            language: Язык для обработки ('english' или 'russian')
        """
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        
        # Загрузка необходимых ресурсов NLTK
        self._ensure_nltk_data()
        
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words(language))
        
        # Для русского языка используем pymorphy3
        if language == 'russian':
            try:
                import pymorphy3
                self.morph = pymorphy3.MorphAnalyzer()
            except ImportError:
                print("Warning: pymorphy3 not installed. Russian lemmatization will not work.")
                self.morph = None
    
    def _ensure_nltk_data(self):
        """Проверка и загрузка необходимых данных NLTK."""
        required_data = [
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
        ]
        
        for path, name in required_data:
            try:
                nltk.data.find(path)
            except LookupError:
                print(f"Загрузка NLTK данных: {name}...")
                nltk.download(name, quiet=True)
    
    def clean_text(self, text: str) -> str:
        """
        Очистка текста от специальных символов и лишних пробелов.
        
        Args:
            text: Исходный текст
            
        Returns:
            Очищенный текст
        """
        # Удаление URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Удаление email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Удаление цифр (опционально)
        # text = re.sub(r'\d+', '', text)
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Токенизация текста.
        
        Args:
            text: Исходный текст
            
        Returns:
            Список токенов
        """
        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            nltk.download('punkt')
            tokens = word_tokenize(text.lower())
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Удаление стоп-слов из списка токенов.
        
        Args:
            tokens: Список токенов
            
        Returns:
            Список токенов без стоп-слов
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def remove_punctuation(self, tokens: List[str]) -> List[str]:
        """
        Удаление пунктуации из списка токенов.
        
        Args:
            tokens: Список токенов
            
        Returns:
            Список токенов без пунктуации
        """
        return [token for token in tokens if token not in string.punctuation]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Лемматизация токенов.
        
        Args:
            tokens: Список токенов
            
        Returns:
            Список лемматизированных токенов
        """
        if self.language == 'russian' and self.morph:
            return [self.morph.parse(token)[0].normal_form for token in tokens]
        else:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def process(self, text: str, remove_stopwords: bool = True, 
                lemmatize: bool = True) -> List[str]:
        """
        Полная обработка текста.
        
        Args:
            text: Исходный текст
            remove_stopwords: Удалять ли стоп-слова
            lemmatize: Применять ли лемматизацию
            
        Returns:
            Список обработанных токенов
        """
        # Очистка текста
        text = self.clean_text(text)
        
        # Токенизация
        tokens = self.tokenize(text)
        
        # Удаление пунктуации
        tokens = self.remove_punctuation(tokens)
        
        # Удаление стоп-слов
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # Лемматизация
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        # Фильтрация пустых токенов
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def get_ngrams(self, tokens: List[str], n: int = 2) -> List[tuple]:
        """
        Получение n-грамм из списка токенов.
        
        Args:
            tokens: Список токенов
            n: Размер n-грамм
            
        Returns:
            Список n-грамм
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def get_word_frequency(self, tokens: List[str]) -> Counter:
        """
        Подсчет частоты слов.
        
        Args:
            tokens: Список токенов
            
        Returns:
            Counter с частотами слов
        """
        return Counter(tokens)

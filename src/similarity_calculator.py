"""
Модуль для вычисления схожести текстов различными методами.
"""
import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from collections import Counter


class SimilarityCalculator:
    """Класс для вычисления схожести между текстами."""
    
    def __init__(self):
        """Инициализация калькулятора схожести."""
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
    
    def cosine_similarity_tfidf(self, documents: List[str], 
                                 min_df: int = 1, 
                                 max_df: float = 1.0) -> np.ndarray:
        """
        Вычисление cosine similarity с использованием TF-IDF.
        
        Args:
            documents: Список документов (текстов)
            min_df: Минимальная частота документа
            max_df: Максимальная частота документа
            
        Returns:
            Матрица схожести (numpy array)
        """
        # Создание TF-IDF векторизатора
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            ngram_range=(1, 2)  # unigrams and bigrams
        )
        
        # Преобразование документов в TF-IDF матрицу
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        
        # Вычисление cosine similarity
        similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        return similarity_matrix
    
    def cosine_similarity_tokens(self, tokens_list: List[List[str]]) -> np.ndarray:
        """
        Вычисление cosine similarity на основе токенов.
        
        Args:
            tokens_list: Список списков токенов для каждого документа
            
        Returns:
            Матрица схожести
        """
        # Преобразование токенов обратно в строки
        documents = [' '.join(tokens) for tokens in tokens_list]
        return self.cosine_similarity_tfidf(documents)
    
    def longest_common_subsequence(self, text1: str, text2: str) -> Tuple[int, float]:
        """
        Вычисление longest common subsequence (LCS).
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            Кортеж (длина LCS, процент схожести)
        """
        matcher = SequenceMatcher(None, text1, text2)
        lcs_length = sum(block.size for block in matcher.get_matching_blocks())
        
        # Процент схожести относительно среднего размера текстов
        avg_length = (len(text1) + len(text2)) / 2
        similarity_percent = (lcs_length / avg_length * 100) if avg_length > 0 else 0
        
        return lcs_length, similarity_percent
    
    def sequence_matcher_ratio(self, text1: str, text2: str) -> float:
        """
        Вычисление схожести с помощью SequenceMatcher.
        
        Args:
            text1: Первый текст
            text2: Второй текст
            
        Returns:
            Коэффициент схожести (0-1)
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def ngram_similarity(self, tokens1: List[str], tokens2: List[str], n: int = 2) -> float:
        """
        Вычисление схожести на основе n-грамм.
        
        Args:
            tokens1: Токены первого документа
            tokens2: Токены второго документа
            n: Размер n-грамм
            
        Returns:
            Коэффициент схожести (0-1)
        """
        # Создание n-грамм
        ngrams1 = self._get_ngrams(tokens1, n)
        ngrams2 = self._get_ngrams(tokens2, n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        # Подсчет пересечения
        common_ngrams = ngrams1 & ngrams2
        
        # Jaccard similarity
        jaccard = len(common_ngrams) / len(ngrams1 | ngrams2)
        
        return jaccard
    
    def _get_ngrams(self, tokens: List[str], n: int) -> set:
        """
        Получение множества n-грамм из токенов.
        
        Args:
            tokens: Список токенов
            n: Размер n-грамм
            
        Returns:
            Множество n-грамм
        """
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    
    def calculate_all_similarities(self, text1: str, text2: str, 
                                   tokens1: List[str], tokens2: List[str]) -> Dict[str, float]:
        """
        Вычисление всех метрик схожести между двумя текстами.
        
        Args:
            text1: Первый текст
            text2: Второй текст
            tokens1: Токены первого текста
            tokens2: Токены второго текста
            
        Returns:
            Словарь с различными метриками схожести
        """
        results = {}
        
        # Cosine similarity (TF-IDF)
        cos_sim_matrix = self.cosine_similarity_tfidf([text1, text2])
        results['cosine_tfidf'] = cos_sim_matrix[0][1]
        
        # Sequence matcher ratio
        results['sequence_matcher'] = self.sequence_matcher_ratio(text1, text2)
        
        # LCS
        lcs_length, lcs_percent = self.longest_common_subsequence(text1, text2)
        results['lcs_length'] = lcs_length
        results['lcs_percent'] = lcs_percent
        
        # N-gram similarities
        results['bigram_similarity'] = self.ngram_similarity(tokens1, tokens2, n=2)
        results['trigram_similarity'] = self.ngram_similarity(tokens1, tokens2, n=3)
        
        # Вычисление средней схожести
        metrics = [
            results['cosine_tfidf'],
            results['sequence_matcher'],
            results['lcs_percent'] / 100,
            results['bigram_similarity'],
            results['trigram_similarity']
        ]
        results['average_similarity'] = np.mean(metrics)
        
        return results
    
    def compare_multiple_documents(self, documents: Dict[str, str], 
                                   processed_tokens: Dict[str, List[str]]) -> Dict:
        """
        Сравнение нескольких документов между собой.
        
        Args:
            documents: Словарь {имя_документа: текст}
            processed_tokens: Словарь {имя_документа: список_токенов}
            
        Returns:
            Словарь с результатами сравнения
        """
        doc_names = list(documents.keys())
        n_docs = len(doc_names)
        
        # Матрицы для разных метрик
        cosine_matrix = np.zeros((n_docs, n_docs))
        sequence_matrix = np.zeros((n_docs, n_docs))
        bigram_matrix = np.zeros((n_docs, n_docs))
        average_matrix = np.zeros((n_docs, n_docs))
        
        # Вычисление cosine similarity для всех документов сразу
        texts = [documents[name] for name in doc_names]
        cosine_matrix = self.cosine_similarity_tfidf(texts)
        
        # Вычисление остальных метрик попарно
        for i in range(n_docs):
            for j in range(n_docs):
                if i == j:
                    sequence_matrix[i][j] = 1.0
                    bigram_matrix[i][j] = 1.0
                    average_matrix[i][j] = 1.0
                elif i < j:  # Вычисляем только для верхнего треугольника
                    # Sequence matcher
                    sequence_matrix[i][j] = self.sequence_matcher_ratio(
                        documents[doc_names[i]], 
                        documents[doc_names[j]]
                    )
                    sequence_matrix[j][i] = sequence_matrix[i][j]
                    
                    # Bigram similarity
                    bigram_matrix[i][j] = self.ngram_similarity(
                        processed_tokens[doc_names[i]],
                        processed_tokens[doc_names[j]],
                        n=2
                    )
                    bigram_matrix[j][i] = bigram_matrix[i][j]
                    
                    # Average
                    average_matrix[i][j] = np.mean([
                        cosine_matrix[i][j],
                        sequence_matrix[i][j],
                        bigram_matrix[i][j]
                    ])
                    average_matrix[j][i] = average_matrix[i][j]
        
        return {
            'document_names': doc_names,
            'cosine_similarity': cosine_matrix,
            'sequence_matcher': sequence_matrix,
            'bigram_similarity': bigram_matrix,
            'average_similarity': average_matrix
        }

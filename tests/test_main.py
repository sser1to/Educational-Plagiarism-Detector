"""
Тесты для главного модуля детектора плагиата.
"""
import pytest
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import PlagiarismDetector, analyze_plagiarism
from src.text_processor import TextProcessor
from src.document_loader import DocumentLoader
from src.similarity_calculator import SimilarityCalculator


class TestTextProcessor:
    """Тесты для класса TextProcessor."""
    
    def test_clean_text(self):
        """Тест очистки текста."""
        processor = TextProcessor()
        text = "Check out https://example.com and email@test.com    with   spaces"
        cleaned = processor.clean_text(text)
        
        assert "https://example.com" not in cleaned
        assert "email@test.com" not in cleaned
        assert "  " not in cleaned  # Двойные пробелы удалены
    
    def test_tokenize(self):
        """Тест токенизации."""
        processor = TextProcessor()
        text = "Hello world! This is a test."
        tokens = processor.tokenize(text)
        
        assert len(tokens) > 0
        assert "hello" in tokens
        assert "world" in tokens
    
    def test_remove_stopwords(self):
        """Тест удаления стоп-слов."""
        processor = TextProcessor()
        tokens = ["the", "quick", "brown", "fox", "is", "a", "test"]
        filtered = processor.remove_stopwords(tokens)
        
        assert "the" not in filtered
        assert "is" not in filtered
        assert "quick" in filtered
        assert "fox" in filtered
    
    def test_lemmatize(self):
        """Тест лемматизации."""
        processor = TextProcessor()
        tokens = ["running", "runs", "ran"]
        lemmatized = processor.lemmatize(tokens)
        
        # Все формы должны привестись к базовой
        assert len(set(lemmatized)) <= len(tokens)
    
    def test_process(self):
        """Тест полной обработки текста."""
        processor = TextProcessor()
        text = "The running foxes are jumping over the fence!"
        tokens = processor.process(text)
        
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
        assert "the" not in tokens  # стоп-слова удалены
    
    def test_get_ngrams(self):
        """Тест получения n-грамм."""
        processor = TextProcessor()
        tokens = ["this", "is", "a", "test"]
        bigrams = processor.get_ngrams(tokens, n=2)
        
        assert len(bigrams) == 3
        assert ("this", "is") in bigrams
        assert ("is", "a") in bigrams
        assert ("a", "test") in bigrams


class TestDocumentLoader:
    """Тесты для класса DocumentLoader."""
    
    def test_load_txt(self, tmp_path):
        """Тест загрузки текстового файла."""
        # Создание временного файла
        test_file = tmp_path / "test.txt"
        test_content = "This is a test file."
        test_file.write_text(test_content, encoding='utf-8')
        
        loader = DocumentLoader()
        content = loader.load(str(test_file))
        
        assert content == test_content
    
    def test_load_nonexistent_file(self):
        """Тест загрузки несуществующего файла."""
        loader = DocumentLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_file.txt")
    
    def test_load_unsupported_format(self, tmp_path):
        """Тест загрузки неподдерживаемого формата."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")
        
        loader = DocumentLoader()
        
        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            loader.load(str(test_file))
    
    def test_load_multiple(self, tmp_path):
        """Тест загрузки нескольких файлов."""
        # Создание нескольких файлов
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")
        
        loader = DocumentLoader()
        documents = loader.load_multiple([str(file1), str(file2)])
        
        assert len(documents) == 2
        assert "test1.txt" in documents
        assert "test2.txt" in documents


class TestSimilarityCalculator:
    """Тесты для класса SimilarityCalculator."""
    
    def test_cosine_similarity_identical(self):
        """Тест cosine similarity для идентичных текстов."""
        calculator = SimilarityCalculator()
        documents = ["This is a test", "This is a test"]
        
        similarity = calculator.cosine_similarity_tfidf(documents)
        
        assert similarity[0][1] == pytest.approx(1.0, abs=0.01)
    
    def test_cosine_similarity_different(self):
        """Тест cosine similarity для разных текстов."""
        calculator = SimilarityCalculator()
        documents = ["This is about cats", "This is about dogs"]
        
        similarity = calculator.cosine_similarity_tfidf(documents)
        
        assert 0 < similarity[0][1] < 1
    
    def test_sequence_matcher_ratio(self):
        """Тест SequenceMatcher."""
        calculator = SimilarityCalculator()
        
        # Идентичные тексты
        ratio1 = calculator.sequence_matcher_ratio("test", "test")
        assert ratio1 == 1.0
        
        # Полностью разные тексты
        ratio2 = calculator.sequence_matcher_ratio("abc", "xyz")
        assert ratio2 < 0.5
    
    def test_ngram_similarity(self):
        """Тест n-gram similarity."""
        calculator = SimilarityCalculator()
        tokens1 = ["this", "is", "a", "test"]
        tokens2 = ["this", "is", "a", "test"]
        
        similarity = calculator.ngram_similarity(tokens1, tokens2, n=2)
        assert similarity == 1.0
    
    def test_calculate_all_similarities(self):
        """Тест вычисления всех метрик."""
        calculator = SimilarityCalculator()
        text1 = "This is the first test document"
        text2 = "This is the second test document"
        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()
        
        results = calculator.calculate_all_similarities(text1, text2, tokens1, tokens2)
        
        assert 'cosine_tfidf' in results
        assert 'sequence_matcher' in results
        assert 'bigram_similarity' in results
        assert 'average_similarity' in results
        assert 0 <= results['average_similarity'] <= 1


class TestPlagiarismDetector:
    """Тесты для класса PlagiarismDetector."""
    
    def test_init(self):
        """Тест инициализации детектора."""
        detector = PlagiarismDetector()
        
        assert detector.language == 'english'
        assert detector.loader is not None
        assert detector.processor is not None
        assert detector.calculator is not None
    
    def test_load_documents(self, tmp_path):
        """Тест загрузки документов."""
        # Создание тестовых файлов
        file1 = tmp_path / "doc1.txt"
        file2 = tmp_path / "doc2.txt"
        file1.write_text("Document 1 content")
        file2.write_text("Document 2 content")
        
        detector = PlagiarismDetector()
        documents = detector.load_documents([str(file1), str(file2)])
        
        assert len(documents) == 2
        assert "doc1.txt" in documents
        assert "doc2.txt" in documents
    
    def test_process_documents(self, tmp_path):
        """Тест обработки документов."""
        # Создание тестовых файлов
        file1 = tmp_path / "doc1.txt"
        file1.write_text("This is a test document with some words")
        
        detector = PlagiarismDetector()
        detector.load_documents([str(file1)])
        tokens = detector.process_documents()
        
        assert len(tokens) == 1
        assert "doc1.txt" in tokens
        assert len(tokens["doc1.txt"]) > 0
    
    def test_compare_documents(self, tmp_path):
        """Тест сравнения документов."""
        # Создание тестовых файлов
        file1 = tmp_path / "doc1.txt"
        file2 = tmp_path / "doc2.txt"
        file1.write_text("This is document one with specific content")
        file2.write_text("This is document two with different content")
        
        detector = PlagiarismDetector()
        detector.load_documents([str(file1), str(file2)])
        detector.process_documents()
        results = detector.compare_documents()
        
        assert 'document_names' in results
        assert 'cosine_similarity' in results
        assert 'average_similarity' in results
        assert len(results['document_names']) == 2
    
    def test_generate_report(self, tmp_path):
        """Тест генерации отчета."""
        # Создание тестовых файлов
        file1 = tmp_path / "doc1.txt"
        file2 = tmp_path / "doc2.txt"
        file1.write_text("The quick brown fox jumps over the lazy dog")
        file2.write_text("The quick brown fox jumps over the lazy dog")  # Идентичный
        
        detector = PlagiarismDetector()
        detector.load_documents([str(file1), str(file2)])
        detector.process_documents()
        detector.compare_documents()
        
        report = detector.generate_report(threshold=0.7)
        
        assert 'total_documents' in report
        assert 'suspicious_pairs' in report
        assert report['total_documents'] == 2
        # Идентичные документы должны быть обнаружены
        assert len(report['suspicious_pairs']) > 0


def test_analyze_plagiarism_function(tmp_path):
    """Тест главной функции analyze_plagiarism."""
    # Создание тестовых файлов
    file1 = tmp_path / "doc1.txt"
    file2 = tmp_path / "doc2.txt"
    file1.write_text("This is a test document")
    file2.write_text("This is another test document")
    
    detector = analyze_plagiarism(
        [str(file1), str(file2)],
        visualize=False
    )
    
    assert detector is not None
    assert len(detector.documents) == 2
    assert detector.comparison_results is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

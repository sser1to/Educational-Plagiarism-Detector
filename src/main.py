"""
Главный модуль для детектирования плагиата в студенческих работах.
"""
import os
from typing import List, Dict, Optional, Union
from pathlib import Path

from .document_loader import DocumentLoader
from .text_processor import TextProcessor
from .similarity_calculator import SimilarityCalculator
from .visualizer import SimilarityVisualizer


class PlagiarismDetector:
    """
    Основной класс для детектирования плагиата.
    
    Объединяет все компоненты системы: загрузку документов,
    обработку текста, вычисление схожести и визуализацию.
    """
    
    def __init__(self, language: str = 'english'):
        """
        Инициализация детектора плагиата.
        
        Args:
            language: Язык для обработки текста ('english' или 'russian')
        """
        self.language = language
        self.loader = DocumentLoader()
        self.processor = TextProcessor(language=language)
        self.calculator = SimilarityCalculator()
        self.visualizer = SimilarityVisualizer()
        
        self.documents = {}
        self.processed_tokens = {}
        self.comparison_results = None
    
    def load_documents(self, source: Union[str, List[str]], 
                       recursive: bool = False) -> Dict[str, str]:
        """
        Загрузка документов из файла, списка файлов или директории.
        
        Args:
            source: Путь к файлу, список путей или директория
            recursive: Рекурсивный поиск в поддиректориях (для директорий)
            
        Returns:
            Словарь {имя_файла: содержимое}
        """
        if isinstance(source, str):
            path = Path(source)
            if path.is_dir():
                self.documents = self.loader.load_from_directory(source, recursive)
            elif path.is_file():
                filename = path.name
                self.documents = {filename: self.loader.load(source)}
            else:
                raise ValueError(f"Путь не найден: {source}")
        elif isinstance(source, list):
            self.documents = self.loader.load_multiple(source)
        else:
            raise TypeError("source должен быть строкой или списком строк")
        
        print(f"Загружено документов: {len(self.documents)}")
        return self.documents
    
    def process_documents(self, remove_stopwords: bool = True, 
                          lemmatize: bool = True) -> Dict[str, List[str]]:
        """
        Обработка загруженных документов.
        
        Args:
            remove_stopwords: Удалять ли стоп-слова
            lemmatize: Применять ли лемматизацию
            
        Returns:
            Словарь {имя_файла: список_токенов}
        """
        if not self.documents:
            raise ValueError("Сначала загрузите документы с помощью load_documents()")
        
        print("Обработка документов...")
        for filename, text in self.documents.items():
            tokens = self.processor.process(
                text, 
                remove_stopwords=remove_stopwords,
                lemmatize=lemmatize
            )
            self.processed_tokens[filename] = tokens
        
        print(f"Обработано документов: {len(self.processed_tokens)}")
        return self.processed_tokens
    
    def compare_documents(self) -> Dict:
        """
        Сравнение всех загруженных документов между собой.
        
        Returns:
            Словарь с результатами сравнения
        """
        if not self.documents or not self.processed_tokens:
            raise ValueError("Сначала загрузите и обработайте документы")
        
        print("Сравнение документов...")
        self.comparison_results = self.calculator.compare_multiple_documents(
            self.documents,
            self.processed_tokens
        )
        
        print("Сравнение завершено")
        return self.comparison_results
    
    def compare_two_documents(self, doc1_name: str, doc2_name: str) -> Dict[str, float]:
        """
        Детальное сравнение двух конкретных документов.
        
        Args:
            doc1_name: Имя первого документа
            doc2_name: Имя второго документа
            
        Returns:
            Словарь с метриками схожести
        """
        if doc1_name not in self.documents or doc2_name not in self.documents:
            raise ValueError("Документы не найдены")
        
        results = self.calculator.calculate_all_similarities(
            self.documents[doc1_name],
            self.documents[doc2_name],
            self.processed_tokens[doc1_name],
            self.processed_tokens[doc2_name]
        )
        
        return results
    
    def visualize_results(self, save_dir: Optional[str] = None,
                         threshold: float = 0.7) -> None:
        """
        Визуализация результатов сравнения.
        
        Args:
            save_dir: Директория для сохранения графиков (если None, показываются)
            threshold: Порог для определения высокой схожести
        """
        if not self.comparison_results:
            raise ValueError("Сначала выполните сравнение документов")
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Создание комплексного отчета
        report_path = os.path.join(save_dir, 'plagiarism_report.png') if save_dir else None
        self.visualizer.plot_plagiarism_report(
            self.comparison_results,
            threshold=threshold,
            save_path=report_path
        )
        
        # Создание матриц сравнения
        matrices_path = save_dir if save_dir else None
        self.visualizer.plot_multiple_matrices(
            self.comparison_results,
            save_dir=matrices_path
        )
    
    def generate_report(self, threshold: float = 0.7) -> Dict:
        """
        Генерация текстового отчета о плагиате.
        
        Args:
            threshold: Порог для определения высокой схожести
            
        Returns:
            Словарь с информацией о подозрительных парах
        """
        if not self.comparison_results:
            raise ValueError("Сначала выполните сравнение документов")
        
        doc_names = self.comparison_results['document_names']
        avg_matrix = self.comparison_results['average_similarity']
        n_docs = len(doc_names)
        
        suspicious_pairs = []
        for i in range(n_docs):
            for j in range(i+1, n_docs):
                similarity = avg_matrix[i][j]
                if similarity >= threshold:
                    suspicious_pairs.append({
                        'document1': doc_names[i],
                        'document2': doc_names[j],
                        'similarity': similarity,
                        'cosine': self.comparison_results['cosine_similarity'][i][j],
                        'sequence_matcher': self.comparison_results['sequence_matcher'][i][j],
                        'bigram': self.comparison_results['bigram_similarity'][i][j]
                    })
        
        # Сортировка по убыванию схожести
        suspicious_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'total_documents': n_docs,
            'suspicious_pairs': suspicious_pairs,
            'threshold': threshold
        }
    
    def print_report(self, threshold: float = 0.7) -> None:
        """
        Вывод отчета о плагиате в консоль.
        
        Args:
            threshold: Порог для определения высокой схожести
        """
        report = self.generate_report(threshold)
        
        print("\n" + "="*70)
        print("ОТЧЕТ О ДЕТЕКТИРОВАНИИ ПЛАГИАТА")
        print("="*70)
        print(f"\nВсего документов: {report['total_documents']}")
        print(f"Порог схожести: {report['threshold']:.1%}")
        print(f"Найдено подозрительных пар: {len(report['suspicious_pairs'])}")
        
        if report['suspicious_pairs']:
            print("\n" + "-"*70)
            print("ПОДОЗРИТЕЛЬНЫЕ ПАРЫ:")
            print("-"*70)
            
            for idx, pair in enumerate(report['suspicious_pairs'], 1):
                risk_level = "🔴 ВЫСОКИЙ" if pair['similarity'] >= 0.9 else "🟡 СРЕДНИЙ"
                print(f"\n{idx}. {risk_level} РИСК ПЛАГИАТА")
                print(f"   Документы:")
                print(f"     - {pair['document1']}")
                print(f"     - {pair['document2']}")
                print(f"   Средняя схожесть: {pair['similarity']:.1%}")
                print(f"   Детали:")
                print(f"     • Cosine (TF-IDF): {pair['cosine']:.1%}")
                print(f"     • Sequence Matcher: {pair['sequence_matcher']:.1%}")
                print(f"     • Bigram: {pair['bigram']:.1%}")
        else:
            print("\n✓ Подозрительных пар не найдено")
        
        print("\n" + "="*70 + "\n")


def analyze_plagiarism(source: Union[str, List[str]],
                       language: str = 'english',
                       threshold: float = 0.7,
                       visualize: bool = True,
                       save_dir: Optional[str] = None) -> PlagiarismDetector:
    """
    Упрощенная функция для анализа плагиата.
    
    Args:
        source: Путь к файлу, список файлов или директория
        language: Язык документов ('english' или 'russian')
        threshold: Порог для определения плагиата
        visualize: Создавать ли визуализации
        save_dir: Директория для сохранения результатов
        
    Returns:
        Объект PlagiarismDetector с результатами
    """
    # Создание детектора
    detector = PlagiarismDetector(language=language)
    
    # Загрузка документов
    detector.load_documents(source)
    
    # Обработка документов
    detector.process_documents()
    
    # Сравнение документов
    detector.compare_documents()
    
    # Вывод отчета
    detector.print_report(threshold=threshold)
    
    # Визуализация
    if visualize:
        detector.visualize_results(save_dir=save_dir, threshold=threshold)
    
    return detector


# Для обратной совместимости с README
def analyze_data(data_path: str) -> dict:
    """
    Простая функция анализа для примера в README.
    
    Args:
        data_path: Путь к данным
        
    Returns:
        Результаты анализа
    """
    detector = analyze_plagiarism(data_path, visualize=False)
    return detector.generate_report()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python -m src.main <путь_к_документам>")
        print("Пример: python -m src.main uploads/")
        sys.exit(1)
    
    source_path = sys.argv[1]
    analyze_plagiarism(source_path, save_dir="results")

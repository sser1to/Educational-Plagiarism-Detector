"""
Пример использования системы детектирования плагиата.

Этот скрипт демонстрирует основные возможности Educational Plagiarism Detector.
"""
import os
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.main import PlagiarismDetector, analyze_plagiarism


def example_1_basic_usage():
    """Пример 1: Базовое использование."""
    print("\n" + "="*70)
    print("ПРИМЕР 1: Базовое использование")
    print("="*70)
    
    # Создание тестовых файлов
    uploads_dir = project_root / "uploads"
    uploads_dir.mkdir(exist_ok=True)
    
    # Создание тестовых документов
    doc1 = uploads_dir / "student1.txt"
    doc2 = uploads_dir / "student2.txt"
    doc3 = uploads_dir / "student3.txt"
    
    doc1.write_text("""
    Machine learning is a subset of artificial intelligence that focuses on the 
    development of algorithms that can learn from and make predictions on data. 
    The key idea is to build systems that can automatically improve their performance 
    through experience without being explicitly programmed.
    """)
    
    doc2.write_text("""
    Machine learning is a subset of artificial intelligence that focuses on the 
    development of algorithms that can learn from data and make predictions. 
    The main concept is to create systems that automatically improve their performance 
    through experience without explicit programming.
    """)
    
    doc3.write_text("""
    Deep learning is a specialized area of machine learning that uses neural networks 
    with multiple layers. These networks can learn hierarchical representations of data, 
    making them particularly effective for complex tasks like image recognition and 
    natural language processing.
    """)
    
    print(f"Создано тестовых документов: 3")
    print(f"Директория: {uploads_dir}")
    
    # Использование упрощенной функции
    detector = analyze_plagiarism(
        str(uploads_dir),
        language='english',
        threshold=0.6,
        visualize=False  # Отключаем визуализацию для примера
    )
    
    print("\n✓ Анализ завершен!")


def example_2_advanced_usage():
    """Пример 2: Расширенное использование с настройками."""
    print("\n" + "="*70)
    print("ПРИМЕР 2: Расширенное использование")
    print("="*70)
    
    # Создание детектора с настройками
    detector = PlagiarismDetector(language='english')
    
    # Загрузка документов
    uploads_dir = project_root / "uploads"
    if not uploads_dir.exists() or not list(uploads_dir.glob("*.txt")):
        print("⚠ Нет документов для анализа. Запустите сначала example_1")
        return
    
    print(f"Загрузка документов из {uploads_dir}...")
    detector.load_documents(str(uploads_dir))
    
    # Обработка с настройками
    print("Обработка текстов...")
    detector.process_documents(
        remove_stopwords=True,
        lemmatize=True
    )
    
    # Сравнение документов
    print("Сравнение документов...")
    results = detector.compare_documents()
    
    # Детальное сравнение двух документов
    doc_names = results['document_names']
    if len(doc_names) >= 2:
        print(f"\nДетальное сравнение: {doc_names[0]} vs {doc_names[1]}")
        detailed = detector.compare_two_documents(doc_names[0], doc_names[1])
        
        print(f"  Cosine TF-IDF: {detailed['cosine_tfidf']:.2%}")
        print(f"  Sequence Matcher: {detailed['sequence_matcher']:.2%}")
        print(f"  Bigram Similarity: {detailed['bigram_similarity']:.2%}")
        print(f"  Средняя схожесть: {detailed['average_similarity']:.2%}")
    
    # Генерация отчета
    print("\nГенерация отчета о плагиате...")
    report = detector.generate_report(threshold=0.6)
    
    print(f"\nНайдено подозрительных пар: {len(report['suspicious_pairs'])}")
    for idx, pair in enumerate(report['suspicious_pairs'][:3], 1):
        print(f"\n  {idx}. {pair['document1']} ↔ {pair['document2']}")
        print(f"     Схожесть: {pair['similarity']:.1%}")


def example_3_visualization():
    """Пример 3: Визуализация результатов."""
    print("\n" + "="*70)
    print("ПРИМЕР 3: Визуализация результатов")
    print("="*70)
    
    uploads_dir = project_root / "uploads"
    if not uploads_dir.exists() or not list(uploads_dir.glob("*.txt")):
        print("⚠ Нет документов для анализа. Запустите сначала example_1")
        return
    
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    print(f"Анализ документов с визуализацией...")
    print(f"Результаты будут сохранены в: {results_dir}")
    
    detector = analyze_plagiarism(
        str(uploads_dir),
        language='english',
        threshold=0.6,
        visualize=True,
        save_dir=str(results_dir)
    )
    
    print(f"\n✓ Графики сохранены в {results_dir}")


def example_4_custom_files():
    """Пример 4: Анализ конкретных файлов."""
    print("\n" + "="*70)
    print("ПРИМЕР 4: Анализ списка файлов")
    print("="*70)
    
    # Создание документов в директории data
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    essay1 = data_dir / "essay1.txt"
    essay2 = data_dir / "essay2.txt"
    
    essay1.write_text("""
    Climate change is one of the most pressing issues of our time. Rising global 
    temperatures are causing ice caps to melt, sea levels to rise, and weather 
    patterns to become more extreme. Scientists agree that human activities, 
    particularly the burning of fossil fuels, are the primary cause.
    """)
    
    essay2.write_text("""
    The impact of climate change on biodiversity is significant. As temperatures 
    rise and habitats change, many species are struggling to adapt. Some are 
    migrating to new areas, while others face extinction. Conservation efforts 
    are critical to protecting vulnerable species.
    """)
    
    print(f"Создано тестовых эссе: 2")
    
    # Анализ конкретных файлов
    detector = PlagiarismDetector()
    detector.load_documents([str(essay1), str(essay2)])
    detector.process_documents()
    detector.compare_documents()
    detector.print_report(threshold=0.5)


def main():
    """Главная функция с меню выбора примеров."""
    print("\n" + "="*70)
    print("Educational Plagiarism Detector - Примеры использования")
    print("="*70)
    
    examples = {
        '1': ('Базовое использование', example_1_basic_usage),
        '2': ('Расширенное использование', example_2_advanced_usage),
        '3': ('Визуализация результатов', example_3_visualization),
        '4': ('Анализ конкретных файлов', example_4_custom_files),
        'all': ('Запустить все примеры', None)
    }
    
    print("\nДоступные примеры:")
    for key, (name, _) in examples.items():
        if key != 'all':
            print(f"  {key}. {name}")
    print(f"  all. Запустить все примеры")
    print(f"  q. Выход")
    
    choice = input("\nВыберите пример (1-4, all, q): ").strip().lower()
    
    if choice == 'q':
        print("Выход...")
        return
    elif choice == 'all':
        for key, (name, func) in examples.items():
            if key != 'all' and func:
                try:
                    func()
                except Exception as e:
                    print(f"❌ Ошибка в примере {key}: {e}")
    elif choice in examples and examples[choice][1]:
        try:
            examples[choice][1]()
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    else:
        print("❌ Неверный выбор!")
    
    print("\n" + "="*70)
    print("Примеры завершены!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

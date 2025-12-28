#!/usr/bin/env python
"""
Скрипт быстрого запуска для Educational Plagiarism Detector.

Использование:
    python run.py [путь_к_документам] [опции]

Примеры:
    python run.py uploads/
    python run.py uploads/ --threshold 0.6
    python run.py file1.txt file2.txt --no-viz
"""
import sys
import argparse
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent))

from src.main import analyze_plagiarism


def main():
    """Главная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description='Educational Plagiarism Detector - система детектирования плагиата',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python run.py uploads/                    # Анализ всех файлов в директории
  python run.py file1.txt file2.txt         # Анализ конкретных файлов
  python run.py uploads/ --threshold 0.6    # Установка порога схожести
  python run.py uploads/ --no-viz           # Без визуализации
  python run.py uploads/ --lang russian     # Для русского языка
  python run.py uploads/ -o results/        # Сохранение результатов
        """
    )
    
    parser.add_argument(
        'source',
        nargs='+',
        help='Путь к файлу, директории или список файлов для анализа'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.7,
        help='Порог схожести для определения плагиата (0.0-1.0). По умолчанию: 0.7'
    )
    
    parser.add_argument(
        '-l', '--lang',
        choices=['english', 'russian'],
        default='english',
        help='Язык документов. По умолчанию: english'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Отключить визуализацию (быстрее)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Директория для сохранения результатов. По умолчанию: results/'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Рекурсивный поиск файлов в поддиректориях'
    )
    
    parser.add_argument(
        '--no-stopwords',
        action='store_true',
        help='Не удалять стоп-слова при обработке'
    )
    
    parser.add_argument(
        '--no-lemma',
        action='store_true',
        help='Не применять лемматизацию'
    )
    
    args = parser.parse_args()
    
    # Определение источника
    if len(args.source) == 1:
        source = args.source[0]
    else:
        source = args.source
    
    # Вывод информации
    print("\n" + "="*70)
    print("Educational Plagiarism Detector")
    print("="*70)
    print(f"\nИсточник: {source}")
    print(f"Язык: {args.lang}")
    print(f"Порог схожести: {args.threshold:.1%}")
    print(f"Визуализация: {'Отключена' if args.no_viz else 'Включена'}")
    if args.output:
        print(f"Результаты: {args.output}")
    print()
    
    try:
        # Запуск анализа
        detector = analyze_plagiarism(
            source=source,
            language=args.lang,
            threshold=args.threshold,
            visualize=not args.no_viz,
            save_dir=args.output
        )
        
        print("\n✓ Анализ успешно завершен!")
        
        if args.output:
            print(f"✓ Результаты сохранены в {args.output}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: Файл или директория не найдены")
        print(f"   {str(e)}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Ошибка: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

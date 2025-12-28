"""
Модуль для визуализации результатов сравнения документов.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import os


class SimilarityVisualizer:
    """Класс для визуализации результатов анализа схожести."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Инициализация визуализатора.
        
        Args:
            style: Стиль matplotlib для графиков
        """
        try:
            plt.style.use(style)
        except:
            # Fallback если стиль не найден
            pass
        
        # Настройка для русского языка
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_similarity_matrix(self, similarity_matrix: np.ndarray, 
                               labels: List[str],
                               title: str = "Матрица схожести документов",
                               save_path: Optional[str] = None,
                               figsize: tuple = (10, 8),
                               cmap: str = 'RdYlGn',
                               annot: bool = True) -> None:
        """
        Визуализация матрицы схожести в виде тепловой карты.
        
        Args:
            similarity_matrix: Матрица схожести
            labels: Названия документов
            title: Заголовок графика
            save_path: Путь для сохранения (если None, то показывается)
            figsize: Размер графика
            cmap: Цветовая палитра
            annot: Показывать ли значения в ячейках
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Создание тепловой карты
        sns.heatmap(
            similarity_matrix,
            annot=annot,
            fmt='.2f',
            cmap=cmap,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Схожесть"},
            xticklabels=labels,
            yticklabels=labels,
            vmin=0,
            vmax=1,
            ax=ax
        )
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_comparison_results(self, results: dict,
                                doc1_name: str,
                                doc2_name: str,
                                save_path: Optional[str] = None,
                                figsize: tuple = (12, 6)) -> None:
        """
        Визуализация результатов сравнения двух документов.
        
        Args:
            results: Словарь с результатами из calculate_all_similarities
            doc1_name: Имя первого документа
            doc2_name: Имя второго документа
            save_path: Путь для сохранения
            figsize: Размер графика
        """
        # Подготовка данных
        metrics = [
            'Cosine (TF-IDF)',
            'Sequence Matcher',
            'Bigram',
            'Trigram',
            'Среднее'
        ]
        
        values = [
            results['cosine_tfidf'],
            results['sequence_matcher'],
            results['bigram_similarity'],
            results['trigram_similarity'],
            results['average_similarity']
        ]
        
        # Создание графика
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Барчарт
        colors = ['#ff9999' if v < 0.3 else '#ffcc99' if v < 0.7 else '#99ff99' 
                  for v in values]
        bars = ax1.barh(metrics, values, color=colors)
        ax1.set_xlabel('Коэффициент схожести', fontsize=12)
        ax1.set_title(f'Сравнение: {doc1_name} vs {doc2_name}', 
                      fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.grid(axis='x', alpha=0.3)
        
        # Добавление значений на барах
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{value:.2%}', 
                    ha='left', va='center', fontsize=10)
        
        # Радарный график
        angles = np.linspace(0, 2 * np.pi, len(metrics) - 1, endpoint=False).tolist()
        values_radar = values[:-1]  # Без среднего
        values_radar += values_radar[:1]  # Замыкаем круг
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, values_radar, 'o-', linewidth=2, color='#4CAF50')
        ax2.fill(angles, values_radar, alpha=0.25, color='#4CAF50')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(metrics[:-1], size=10)
        ax2.set_ylim(0, 1)
        ax2.set_title('Профиль схожести', fontsize=12, fontweight='bold', pad=20)
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_multiple_matrices(self, comparison_results: dict,
                              save_dir: Optional[str] = None,
                              figsize: tuple = (14, 12)) -> None:
        """
        Визуализация нескольких матриц схожести на одном графике.
        
        Args:
            comparison_results: Результаты из compare_multiple_documents
            save_dir: Директория для сохранения
            figsize: Размер графика
        """
        doc_names = comparison_results['document_names']
        
        # Создание сетки графиков
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Сравнение документов различными методами', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Настройки для каждого графика
        matrices = [
            ('cosine_similarity', 'Cosine Similarity (TF-IDF)', axes[0, 0]),
            ('sequence_matcher', 'Sequence Matcher', axes[0, 1]),
            ('bigram_similarity', 'Bigram Similarity', axes[1, 0]),
            ('average_similarity', 'Средняя схожесть', axes[1, 1])
        ]
        
        for key, title, ax in matrices:
            matrix = comparison_results[key]
            
            sns.heatmap(
                matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                xticklabels=doc_names,
                yticklabels=doc_names,
                vmin=0,
                vmax=1,
                ax=ax
            )
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'comparison_matrices.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_plagiarism_report(self, comparison_results: dict,
                              threshold: float = 0.7,
                              save_path: Optional[str] = None,
                              figsize: tuple = (14, 8)) -> None:
        """
        Создание отчета о потенциальном плагиате.
        
        Args:
            comparison_results: Результаты из compare_multiple_documents
            threshold: Порог для определения высокой схожести
            save_path: Путь для сохранения
            figsize: Размер графика
        """
        doc_names = comparison_results['document_names']
        avg_matrix = comparison_results['average_similarity']
        n_docs = len(doc_names)
        
        # Поиск пар с высокой схожестью
        suspicious_pairs = []
        for i in range(n_docs):
            for j in range(i+1, n_docs):
                similarity = avg_matrix[i][j]
                if similarity >= threshold:
                    suspicious_pairs.append((i, j, similarity))
        
        # Создание графика
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Главная матрица
        ax1 = fig.add_subplot(gs[:, 0])
        sns.heatmap(
            avg_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            xticklabels=doc_names,
            yticklabels=doc_names,
            vmin=0,
            vmax=1,
            ax=ax1
        )
        ax1.set_title('Матрица средней схожести', fontsize=12, fontweight='bold')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Информация о подозрительных парах
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        report_text = f"Отчет о потенциальном плагиате\n"
        report_text += f"{'='*40}\n\n"
        report_text += f"Порог схожести: {threshold:.1%}\n"
        report_text += f"Найдено подозрительных пар: {len(suspicious_pairs)}\n\n"
        
        if suspicious_pairs:
            report_text += "Подозрительные пары:\n"
            for i, j, sim in sorted(suspicious_pairs, key=lambda x: -x[2])[:5]:
                status = "🔴 ВЫСОКИЙ" if sim >= 0.9 else "🟡 СРЕДНИЙ"
                report_text += f"\n{status}\n"
                report_text += f"  {doc_names[i]}\n"
                report_text += f"  ↔ {doc_names[j]}\n"
                report_text += f"  Схожесть: {sim:.1%}\n"
        else:
            report_text += "\nПодозрительных пар не найдено ✓"
        
        ax2.text(0.05, 0.95, report_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Гистограмма схожести
        ax3 = fig.add_subplot(gs[1, 1])
        similarities = []
        for i in range(n_docs):
            for j in range(i+1, n_docs):
                similarities.append(avg_matrix[i][j])
        
        ax3.hist(similarities, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Порог: {threshold:.1%}')
        ax3.set_xlabel('Коэффициент схожести', fontsize=10)
        ax3.set_ylabel('Количество пар', fontsize=10)
        ax3.set_title('Распределение схожести', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        fig.suptitle('Анализ плагиата студенческих работ', 
                     fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Отчет сохранен: {save_path}")
        else:
            plt.show()
        
        plt.close()

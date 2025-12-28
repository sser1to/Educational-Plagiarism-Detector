"""
Модуль для загрузки документов различных форматов.
"""
import os
from typing import Optional
from pathlib import Path
import PyPDF2
import docx


class DocumentLoader:
    """Класс для загрузки документов различных форматов."""
    
    SUPPORTED_FORMATS = ['.txt', '.pdf', '.docx', '.doc']
    
    @staticmethod
    def load_txt(file_path: str) -> str:
        """
        Загрузка текстового файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Содержимое файла в виде строки
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Попытка открыть с другой кодировкой
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """
        Загрузка PDF файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Содержимое файла в виде строки
        """
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Ошибка при чтении PDF файла: {str(e)}")
        
        return text.strip()
    
    @staticmethod
    def load_docx(file_path: str) -> str:
        """
        Загрузка DOCX файла.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Содержимое файла в виде строки
        """
        try:
            doc = docx.Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text)
        except Exception as e:
            raise ValueError(f"Ошибка при чтении DOCX файла: {str(e)}")
    
    @classmethod
    def load(cls, file_path: str) -> str:
        """
        Автоматическая загрузка файла на основе расширения.
        
        Args:
            file_path: Путь к файлу
            
        Returns:
            Содержимое файла в виде строки
            
        Raises:
            ValueError: Если формат файла не поддерживается
            FileNotFoundError: Если файл не найден
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in cls.SUPPORTED_FORMATS:
            raise ValueError(
                f"Неподдерживаемый формат файла: {extension}. "
                f"Поддерживаемые форматы: {', '.join(cls.SUPPORTED_FORMATS)}"
            )
        
        if extension == '.txt':
            return cls.load_txt(file_path)
        elif extension == '.pdf':
            return cls.load_pdf(file_path)
        elif extension in ['.docx', '.doc']:
            return cls.load_docx(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат: {extension}")
    
    @classmethod
    def load_multiple(cls, file_paths: list) -> dict:
        """
        Загрузка нескольких файлов.
        
        Args:
            file_paths: Список путей к файлам
            
        Returns:
            Словарь {имя_файла: содержимое}
        """
        documents = {}
        for file_path in file_paths:
            try:
                filename = Path(file_path).name
                documents[filename] = cls.load(file_path)
            except Exception as e:
                print(f"Ошибка при загрузке {file_path}: {str(e)}")
        
        return documents
    
    @classmethod
    def load_from_directory(cls, directory: str, recursive: bool = False) -> dict:
        """
        Загрузка всех поддерживаемых файлов из директории.
        
        Args:
            directory: Путь к директории
            recursive: Рекурсивный поиск в поддиректориях
            
        Returns:
            Словарь {имя_файла: содержимое}
        """
        path = Path(directory)
        
        if not path.exists() or not path.is_dir():
            raise NotADirectoryError(f"Директория не найдена: {directory}")
        
        documents = {}
        
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in cls.SUPPORTED_FORMATS:
                try:
                    filename = file_path.name
                    documents[filename] = cls.load(str(file_path))
                except Exception as e:
                    print(f"Ошибка при загрузке {file_path}: {str(e)}")
        
        return documents

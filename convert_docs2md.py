import os
from docx import Document

def docx_to_md():
    # Запрос пути к файлу
    docx_path = input("Введите полный путь к файлу .docx: ").strip()
    
    # Проверка существования файла
    if not os.path.isfile(docx_path):
        print("Файл не найден. Проверьте путь и попробуйте снова.")
        return
    
    # Проверка расширения
    if not docx_path.endswith('.docx'):
        print("Указанный файл не является файлом .docx.")
        return
    
    # Открываем документ
    try:
        doc = Document(docx_path)
    except Exception as e:
        print(f"Ошибка при открытии файла: {e}")
        return

    # Генерация пути для сохранения .md
    output_path = os.path.splitext(docx_path)[0] + ".md"

    # Конвертация и запись в файл
    try:
        with open(output_path, 'w', encoding='utf-8') as md_file:
            for paragraph in doc.paragraphs:
                md_file.write(paragraph.text + '\n\n')
        print(f"Файл успешно сохранен: {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")
        return

if __name__ == "__main__":
    docx_to_md()

# app_smart_filter.py — умная фильтрация текста
import streamlit as st
import cv2
import easyocr
import pandas as pd
import re
import os
import tempfile
import time
from ultralytics import YOLO
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Пропуски — Умная фильтрация", layout="wide")
st.title("🧠 Умное распознавание пропусков")

# === ГЛОБАЛЬНЫЕ КЕШИ ===
@st.cache_resource
def load_model():
    """Загрузка модели YOLO - ищет модель в нескольких местах"""
    possible_paths = [
        'best.pt',  # В корне репозитория
        'model/best.pt',  # В папке model
        'weights/best.pt',  # В папке weights
        'runs/detect/propuska_detector5/weights/best.pt',  # Относительный путь
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                st.sidebar.success(f"Модель найдена: {model_path}")
                return YOLO(model_path)
            except Exception as e:
                st.sidebar.warning(f"Ошибка загрузки {model_path}: {e}")
                continue
    
    # Если модель не найдена, возвращаем None
    st.sidebar.error("⚠️ Модель не найдена! Загрузите файл модели.")
    return None

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ru'], gpu=False)

model = load_model()
reader = load_ocr()

# === ЗАГРУЗКА МОДЕЛИ ЧЕРЕЗ ИНТЕРФЕЙС ===
if model is None:
    st.warning("""
    ⚠️ Модель YOLO не найдена!
    
    **Для работы приложения:**
    1. Добавьте файл `best.pt` в ваш репозиторий на Gitflic
    2. Или загрузите модель через интерфейс ниже
    """)
    
    uploaded_model = st.sidebar.file_uploader(
        "📁 Загрузите модель (best.pt)",
        type=['pt'],
        help="Загрузите файл модели YOLO"
    )
    
    if uploaded_model:
        # Сохраняем временно
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
            tmp.write(uploaded_model.getvalue())
            model_path = tmp.name
        
        try:
            model = YOLO(model_path)
            st.sidebar.success("✅ Модель загружена успешно!")
        except Exception as e:
            st.sidebar.error(f"❌ Ошибка загрузки модели: {e}")
            model = None
else:
    st.sidebar.success("✅ Модель YOLO загружена")

# === УМНАЯ ФИЛЬТРАЦИЯ ТЕКСТА (остаётся без изменений) ===
class NameFilter:
    # ... (Ваш существующий код NameFilter остается без изменений)
    
    @staticmethod
    def is_stop_word(word):
        # ... ваш код ...
        pass
    
    @staticmethod
    def is_likely_name_part(word):
        # ... ваш код ...
        pass
    
    @staticmethod
    def extract_fio_from_lines(lines):
        # ... ваш код ...
        pass

# === УЛУЧШЕННАЯ ОБРАБОТКА (остаётся без изменений) ===
def preprocess_for_ocr(image):
    # ... ваш код ...
    pass

def extract_text_with_context(card_image):
    # ... ваш код ...
    pass

# === ОСНОВНАЯ ОБРАБОТКА ===
def process_single_image(image, filename, show_debug=True):
    """Обрабатывает одно изображение"""
    # Проверяем, что модель загружена
    if model is None:
        if show_debug:
            st.error("❌ Модель не загружена!")
        return [], {}
    
    results = []
    debug_info = {}
    
    if show_debug:
        st.subheader(f"📷 Обработка: {filename}")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption=f"Исходное", use_container_width=True)
    
    # Детекция пропусков
    try:
        yolo_results = model(image, conf=0.2, verbose=False)
    except Exception as e:
        if show_debug:
            st.error(f"❌ Ошибка детекции: {e}")
        return [], {'error': str(e)}
    
    # ... остальной код функции остается без изменений ...
    # (ваш существующий код от if show_debug: до конца функции)

# === ИНТЕРФЕЙС STREAMLIT ===
st.sidebar.header("⚙️ Настройки")
debug_mode = st.sidebar.checkbox("Показать отладку", True)

# Показываем предупреждение, если нет модели
if model is None:
    st.error("""
    🚫 **Приложение не готово к работе!**
    
    Для начала работы необходимо:
    1. Добавить файл модели `best.pt` в репозиторий
    2. Или загрузить его через форму выше
    3. После загрузки перезагрузите страницу (F5)
    """)
else:
    conf_threshold = st.sidebar.slider("Порог уверенности YOLO", 0.1, 0.9, 0.2, 0.05)
    
    uploaded_files = st.file_uploader(
        "📷 Загрузите фото с пропусками",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Рекомендуется загружать чёткие фото"
    )
    
    if uploaded_files and model:
        # ... ваш существующий код обработки файлов ...
        # (от all_fios = [] до конца блока)

# === ИНФОРМАЦИЯ О СИСТЕМЕ ===
with st.sidebar.expander("ℹ️ О системе"):
    st.markdown("""
    ### Умная фильтрация включает:
    
    **Фильтрация стоп-слов:**
    - Организации, должности
    - Предлоги, союзы
    - Технические надписи
    
    **Распознавание имён:**
    - База распространённых имён/фамилий
    - Паттерны окончаний
    - Проверка регистра
    
    **Предобработка:**
    - Увеличение контраста
    - Удаление шума
    - Масштабирование для OCR
    
    ---
    
    **Для хостинга на Streamlit Cloud:**
    1. Добавьте `best.pt` в репозиторий
    2. Убедитесь в файле `requirements.txt`
    3. Укажите Python 3.9+ в `runtime.txt`
    """)

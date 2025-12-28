# app.py — для Streamlit Cloud (мобильная версия, без синих фото)
import streamlit as st
import cv2
import easyocr
import pandas as pd
import re
import io
import numpy as np
from ultralytics import YOLO
from collections import Counter

st.set_page_config(page_title="Пропуски — Умная фильтрация", layout="centered")  # ← layout="centered" лучше для мобильных
st.title("🧠 Умное распознавание пропусков")

# === КЕШИ ===
@st.cache_resource
def load_model():
    model_path = 'best.pt'  # ✅ Исправлено: относительный путь!
    return YOLO(model_path)

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ru'], gpu=False)

model = load_model()
reader = load_ocr()


# === УМНАЯ ФИЛЬТРАЦИЯ ===
class NameFilter:
    STOP_WORDS = {
        "университет", "государственный", "студент", "участник", "сотрудник", "управление",
        "на", "по", "в", "из", "от", "до", "за", "с", "к", "у",
        "и", "или", "но", "а", "же",
        "номер", "пропуск", "карта", "фото", "дата", "выдача",
        "действителен", "подпись", "печать", "организация",
        "ао", "оо", "зао", "пао", "нко", "ип",
    }
    
    NAME_PATTERNS = [
        r'^[А-ЯЁ][а-яё]{1,20}$',
        r'^[А-ЯЁ][а-яё]+[- ][А-ЯЁ][а-яё]+$', 
    ]
    
    COMMON_FIRST_NAMES = {
        "андрей", "алексей", "александр", "артем", "артём", "борис", "вадим",
        "валентин", "валерий", "василий", "виктор", "владимир", "владислав",
        "геннадий", "георгий", "григорий", "даниил", "денис", "дмитрий",
        "евгений", "егор", "иван", "игорь", "кирилл", "константин",
        "лев", "леонид", "максим", "михаил", "николай", "олег",
        "павел", "петр", "пётр", "роман", "руслан", "сергей",
        "станислав", "степан", "тимофей", "федор", "фёдор", "юрий",
        "ярослав",
        "алена", "алёна", "алина", "алла", "анастасия", "ангелина",
        "анна", "антонина", "валентина", "валерия", "вера", "виктория",
        "галина", "дарья", "диана", "евгения", "екатерина", "елена",
        "елизавета", "жинар", "зинаида", "инна", "ирина", "кристина",
        "ксения", "ксёния", "лариса", "любовь", "людмила", "марина",
        "мария", "маргарита", "надежда", "наталья", "наталия", "оксана",
        "ольга", "полина", "светлана", "софия", "софья", "тамара",
        "татьяна", "юлия", "яна",
    }
    
    COMMON_LAST_NAMES = { ваш список фамилий без изменений }

    @staticmethod
    def is_stop_word(word):
        word_lower = word.lower()
        if len(word) < 2: return True
        if re.match(r'^\d+$', word): return True
        if re.match(r'^[^а-яА-ЯёЁ]+$', word): return True
        if word_lower in NameFilter.STOP_WORDS: return True
        if word.isupper() and len(word) > 3: return True
        if re.match(r'.*универ.*', word_lower): return True
        return False

    @staticmethod
    def is_likely_name_part(word):
        if NameFilter.is_stop_word(word): return False
        word_lower = word.lower()
        for pattern in NameFilter.NAME_PATTERNS:
            if re.match(pattern, word): return True
        if (word_lower in NameFilter.COMMON_FIRST_NAMES or 
            word_lower in NameFilter.COMMON_LAST_NAMES): return True
        if (word[0].isupper() and len(word) >= 3 and word.isalpha() and
            not any(ch.isdigit() for ch in word)):
            endings = ['ов', 'ев', 'ин', 'ын', 'ова', 'ева', 'ина', 'ына',
                       'ий', 'ой', 'ая', 'яя', 'ль', 'дра', 'ла', 'та']
            for ending in endings:
                if word_lower.endswith(ending):
                    return True
        return False

    @staticmethod
    def extract_fio_from_lines(lines):
        all_words = []
        for line in lines:
            words = re.split(r'[\s,.;:]+', line.strip())
            all_words.extend(words)
        candidates = [w for w in all_words if NameFilter.is_likely_name_part(w)]
        if len(candidates) >= 2:
            fio_parts = []
            for w in candidates:
                if w not in fio_parts:
                    fio_parts.append(w)
            return " ".join(fio_parts[:3])
        return None


# === OCR и обработка ===
def preprocess_for_ocr(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    scale = 1.5
    new_size = (int(binary.shape[1] * scale), int(binary.shape[0] * scale))
    resized = cv2.resize(binary, new_size, interpolation=cv2.INTER_CUBIC)
    return resized


def extract_text_with_context(card_image):
    processed = preprocess_for_ocr(card_image)
    try:
        detailed_results = reader.readtext(
            processed, detail=1, paragraph=False, width_ths=0.7, ycenter_ths=0.5
        )
        all_texts = [text.strip() for _, text, conf in detailed_results if conf > 0.1]
        fio = NameFilter.extract_fio_from_lines(all_texts)
        if fio:
            return fio, all_texts
        likely_names = []
        for text in all_texts:
            words = text.split()
            name_words = [w for w in words if NameFilter.is_likely_name_part(w)]
            likely_names.extend(name_words)
        if len(likely_names) >= 2:
            return " ".join(likely_names[:3]), all_texts
        return None, all_texts
    except Exception as e:
        st.warning(f"Ошибка OCR: {e}")
        return None, []


# === ОБРАБОТКА ОДНОГО ИЗОБРАЖЕНИЯ (с исправлением цвета) ===
def process_single_image_and_display(image, filename, show_debug=True):
    results = []
    
    if show_debug:
        st.subheader(f"📷 {filename}")
        col1, col2 = st.columns(2)
        with col1:
            # ✅ BGR → RGB (фото не синие!)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Исходное", use_container_width=True)
    
    yolo_results = model(image, conf=0.4, verbose=False)
    
    if show_debug and hasattr(yolo_results[0], 'plot'):
        with col2:
            plotted = yolo_results[0].plot()
            # ✅ BGR → RGB
            plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            st.image(plotted_rgb, caption="Детекции", use_container_width=True)
    
    boxes = yolo_results[0].boxes
    cards_found = len(boxes) if boxes is not None else 0
    
    if show_debug:
        st.caption(f"📦 Найдено: {cards_found} пропусков")

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            card = image[y1:y2, x1:x2]
            
            if show_debug:
                with st.expander(f"Пропуск {i+1}"):
                    st.image(cv2.cvtColor(card, cv2.COLOR_BGR2RGB), caption="Вырезанный", use_container_width=True)
            
            fio, all_texts = extract_text_with_context(card)
            
            if fio:
                results.append(fio)
                if show_debug:
                    st.success(f"✅ {fio}")
            elif show_debug:
                st.warning("ФИО не найдено")
    elif show_debug:
        st.warning("❌ Пропуски не найдены")
    
    return results


# === КЭШ ЭКСПОРТА ===
@st.cache_data(ttl=300)
def prepare_export_files(edited_df):
    excel_buffer = io.BytesIO()
    edited_df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_bytes = excel_buffer.getvalue()
    txt_content = "\n".join(
        edited_df["ФИО"]
        .dropna()
        .astype(str)
        .str.strip()
        .where(lambda x: x != "")
        .dropna()
        .tolist()
    )
    return excel_bytes, txt_content


# === ИНТЕРФЕЙС ===
st.sidebar.header("⚙️ Настройки")
debug_mode = st.sidebar.checkbox("Показать отладку", False)  # ← по умолчанию False — чище на телефоне

uploaded_files = st.file_uploader(
    "📸 Загрузите фото пропусков",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Рекомендуется: чёткие фото, пропуски крупно"
)

# Инициализация
if 'all_fios' not in st.session_state:
    st.session_state.all_fios = []
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Обработка — один раз
if uploaded_files and not st.session_state.processed:
    st.session_state.all_fios = []
    
    for idx, uploaded_file in enumerate(uploaded_files):
        file_bytes = uploaded_file.getvalue()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            st.error(f"❗ Не удалось прочитать: {uploaded_file.name}")
            continue
        
        fios = process_single_image_and_display(img, uploaded_file.name, debug_mode)
        st.session_state.all_fios.extend(fios)
    
    st.session_state.processed = True

# === ВЫВОД — всегда, если есть данные ===
if st.session_state.processed:
    all_fios = st.session_state.all_fios
    
    if all_fios:
        # Уникальные, без дублей
        unique_fios = []
        seen = set()
        for fio in all_fios:
            if fio not in seen:
                seen.add(fio)
                unique_fios.append(fio)
        
        st.markdown("---")
        st.subheader("📋 Результаты")
        st.info(f"✅ Найдено: {len(unique_fios)} ФИО")
        
        df_editable = pd.DataFrame(unique_fios, columns=["ФИО"])
        edited_df = st.data_editor(
            df_editable,
            num_rows="dynamic",
            use_container_width=True,
            key="fio_editor"
        )
        
        final_list = edited_df["ФИО"].dropna().astype(str).str.strip()
        final_list = final_list[final_list != ""].tolist()
        
        # Экспорт
        excel_bytes, txt_content = prepare_export_files(edited_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Excel",
                excel_bytes,
                "участники.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with col2:
            st.download_button(
                "📥 TXT",
                txt_content,
                "участники.txt",
                "text/plain"
            )
    
    else:
        st.markdown("---")
        st.subheader("📋 Результаты")
        st.error("❌ Не найдено ни одного ФИО")

# Сброс при новой загрузке
if not uploaded_files:
    st.session_state.processed = False

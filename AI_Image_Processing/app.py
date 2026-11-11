import io
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

st.set_page_config(page_title='AI 影像處理與人臉分析系統', page_icon='📷', layout='wide')
st.markdown('''<style>
.stApp {background-color:#101827;color:#f1f5f9}
h1,h2,h3 {color:#93c5fd}
[data-testid="stMetric"] {background-color:#1e293b;padding:14px;border-radius:12px}
div.stButton>button {border-radius:9px}
</style>''', unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

st.title('📷 AI 互動式影像處理與人臉分析系統')
st.caption('Python × Streamlit × OpenCV｜進階影像處理、人臉分析、直方圖與操作紀錄')
page = st.sidebar.radio('📌 功能頁面', ['影像處理與人臉分析', '歷史紀錄與統計'])

if page == '歷史紀錄與統計':
    st.header('📋 歷史紀錄與統計')
    if not st.session_state.history:
        st.info('目前沒有紀錄。請先處理圖片並按「儲存本次操作紀錄」。')
    else:
        df = pd.DataFrame(st.session_state.history)
        a, b, c = st.columns(3)
        a.metric('操作次數', len(df))
        b.metric('功能種類', df['處理功能'].nunique())
        c.metric('累計偵測人臉', int(df['人臉數量'].sum()))
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.subheader('功能使用次數')
        st.bar_chart(df['處理功能'].value_counts())
        st.download_button('📥 下載歷史紀錄 CSV', df.to_csv(index=False).encode('utf-8-sig'),
                           file_name='image_history.csv', mime='text/csv')
        if st.button('🗑️ 清除全部紀錄'):
            st.session_state.history = []
            st.rerun()
    st.caption('歷史紀錄僅保存在目前 Streamlit 工作階段，不是永久資料庫。')
    st.stop()

operations = ['原始圖片', '灰階處理', '高斯模糊', '邊緣偵測', '亮度與對比度調整',
              '人臉偵測', '圖片旋轉', '圖片縮放', '影像銳化', '中值濾波', '二值化']
operation = st.sidebar.selectbox('⚙️ 選擇處理功能', operations)
show_histogram = st.sidebar.checkbox('顯示 RGB / 灰階直方圖', value=True)
uploaded_file = st.file_uploader('📁 上傳 JPG 或 PNG 圖片', type=['jpg', 'jpeg', 'png'])
if uploaded_file is None:
    st.info('請上傳圖片，然後從左側選擇影像處理功能。')
    st.stop()

try:
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    result = img_array.copy()
    face_data = []
    face_images = []
    face_count = 0

    if operation == '灰階處理':
        result = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    elif operation == '高斯模糊':
        strength = st.sidebar.slider('模糊核心大小', 1, 31, 5, step=2)
        result = cv2.GaussianBlur(img_array, (strength, strength), 0)

    elif operation == '邊緣偵測':
        low = st.sidebar.slider('低閾值', 0, 255, 80)
        high = st.sidebar.slider('高閾值', 0, 255, 160)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        result = cv2.Canny(gray, min(low, high), max(low, high))

    elif operation == '亮度與對比度調整':
        brightness = st.sidebar.slider('亮度', -100, 100, 0)
        contrast = st.sidebar.slider('對比度', 0.5, 3.0, 1.0, step=0.1)
        result = np.clip(img_array.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)

    elif operation == '人臉偵測':
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if detector.empty():
            raise RuntimeError('OpenCV 人臉偵測模型載入失敗')
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_count = len(faces)
        for i, (x, y, w, h) in enumerate(faces, start=1):
            x, y, w, h = map(int, (x, y, w, h))
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(result, f'Face {i}', (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            face_data.append({'人臉編號': i, 'X座標': x, 'Y座標': y, '寬度': w, '高度': h,
                              '面積(px²)': w * h})
            face_images.append(img_array[y:y + h, x:x + w])

    elif operation == '圖片旋轉':
        angle = st.sidebar.slider('旋轉角度', -180, 180, 0)
        h, w = img_array.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        cos_a, sin_a = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_w, new_h = int(h * sin_a + w * cos_a), int(h * cos_a + w * sin_a)
        matrix[0, 2] += new_w / 2 - w / 2
        matrix[1, 2] += new_h / 2 - h / 2
        result = cv2.warpAffine(img_array, matrix, (new_w, new_h),
                                flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    elif operation == '圖片縮放':
        scale = st.sidebar.slider('縮放比例 (%)', 10, 200, 100, step=10)
        new_w = max(1, round(image.width * scale / 100))
        new_h = max(1, round(image.height * scale / 100))
        interpolation = cv2.INTER_AREA if scale < 100 else cv2.INTER_LINEAR
        result = cv2.resize(img_array, (new_w, new_h), interpolation=interpolation)

    elif operation == '影像銳化':
        strength = st.sidebar.slider('銳化強度', 0.0, 3.0, 1.0, step=0.1)
        blurred = cv2.GaussianBlur(img_array, (0, 0), 2.0)
        result = np.clip(img_array.astype(np.float32) * (1 + strength)
                         - blurred.astype(np.float32) * strength, 0, 255).astype(np.uint8)

    elif operation == '中值濾波':
        kernel = st.sidebar.slider('濾波核心大小', 3, 15, 5, step=2)
        result = cv2.medianBlur(img_array, kernel)

    elif operation == '二值化':
        threshold = st.sidebar.slider('二值化閾值', 0, 255, 127)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    st.subheader('🖼️ 原始圖片與處理結果')
    left, right = st.columns(2)
    with left:
        st.image(img_array, caption='原始圖片', use_container_width=True)
    with right:
        st.image(result, caption=f'處理結果：{operation}', use_container_width=True)

    st.subheader('📊 圖片資訊')
    a, b, c, d = st.columns(4)
    a.metric('原圖寬度', f'{image.width} px')
    b.metric('原圖高度', f'{image.height} px')
    c.metric('結果尺寸', f'{result.shape[1]} × {result.shape[0]}')
    d.metric('人臉數量', face_count if operation == '人臉偵測' else '未執行')

    if operation == '人臉偵測':
        st.subheader('👤 人臉分析')
        if face_count:
            st.success(f'偵測到 {face_count} 張人臉')
            st.dataframe(pd.DataFrame(face_data), hide_index=True, use_container_width=True)
            st.write('人臉裁切結果')
            cols = st.columns(min(face_count, 4))
            for i, face in enumerate(face_images):
                with cols[i % len(cols)]:
                    st.image(face, caption=f'人臉 {i + 1}', use_container_width=True)
        else:
            st.warning('沒有偵測到人臉。請嘗試正面、光線充足的照片。')
        st.caption('此功能使用 OpenCV Haar Cascade 偵測人臉位置，不辨識人物身分。')

    if show_histogram:
        st.subheader('📈 處理結果像素直方圖')
        if result.ndim == 2:
            hist = cv2.calcHist([result], [0], None, [256], [0, 256]).ravel()
            chart_df = pd.DataFrame({'灰階': hist}, index=range(256))
        else:
            channels = {}
            for index, label in enumerate(['紅色 R', '綠色 G', '藍色 B']):
                channels[label] = cv2.calcHist([result], [index], None, [256], [0, 256]).ravel()
            chart_df = pd.DataFrame(channels, index=range(256))
        chart_df.index.name = '像素值 (0–255)'
        st.line_chart(chart_df)
        st.caption('橫軸為像素值 0–255，縱軸為該像素值出現的次數。')

    st.divider()
    output = io.BytesIO()
    Image.fromarray(result.astype(np.uint8)).save(output, format='PNG')
    st.download_button('📥 下載處理後圖片', data=output.getvalue(),
                       file_name='processed_image.png', mime='image/png')
    if st.button('💾 儲存本次操作紀錄'):
        st.session_state.history.append({
            '處理時間': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '圖片名稱': uploaded_file.name,
            '處理功能': operation,
            '圖片寬度': image.width,
            '圖片高度': image.height,
            '人臉數量': face_count,
        })
        st.success('紀錄已儲存！請切換左側「歷史紀錄與統計」查看。')

except (UnidentifiedImageError, OSError):
    st.error('無法讀取圖片，請確認檔案是有效的 JPG 或 PNG。')
except Exception as error:
    st.error('處理過程發生錯誤，詳細資訊如下：')
    st.exception(error)

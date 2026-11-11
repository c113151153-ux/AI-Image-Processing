
import streamlit as st
import pandas as pd
from datetime import datetime

def initialize_history():
    if "history" not in st.session_state:
        st.session_state.history = []

def add_history(filename, operation, width, height, face_count=0):
    initialize_history()

    record = {
        "處理時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "圖片名稱": filename,
        "處理功能": operation,
        "圖片寬度": width,
        "圖片高度": height,
        "人臉數量": face_count
    }

    st.session_state.history.append(record)

def show_history():
    initialize_history()

    st.subheader("📋 影像處理歷史紀錄")

    if not st.session_state.history:
        st.info("目前尚無處理紀錄")
        return

    df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📊 操作統計")
    st.bar_chart(df["處理功能"].value_counts())

    csv_data = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "📥 下載歷史紀錄 CSV",
        data=csv_data,
        file_name="image_history.csv",
        mime="text/csv"
    )

    if st.button("🗑️ 清除歷史紀錄"):
        st.session_state.history = []
        st.rerun()

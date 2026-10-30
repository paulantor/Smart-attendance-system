import streamlit as st
import sqlite3
import pandas as pd
import os
from pathlib import Path
from config import DATABASE_DIR, DATASET_DIR, settings

def render():
    st.title("🗂️ Registered Students Database")
    
    # ডাটাবেস থেকে ইনফরমেশন নিয়ে আসা
    try:
        conn = sqlite3.connect(DATABASE_DIR / 'attendance.db')
        # আপনার টেবিলের নাম যদি 'students' না হয়ে অন্য কিছু হয়, তবে নিচের লাইনে সেটি পরিবর্তন করে নেবেন
        df = pd.read_sql_query("SELECT * FROM students", conn)
        conn.close()
    except Exception as e:
        st.error(f"Database error: {e}")
        return

    if df.empty:
        st.info("No students registered yet.")
        return

    # ১. ডাটাবেসের ফুল টেবিল দেখানো
    st.subheader("📋 Student Information Table")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # ২. স্টুডেন্টদের ছবি দেখানো
    st.subheader("🖼️ Registered Faces")
    
    # ফোল্ডার থেকে স্টুডেন্ট আইডি অনুযায়ী ছবি খোঁজা
    dataset_path = DATASET_DIR
    
    if not dataset_path.exists():
        st.warning("Dataset folder not found!")
        return

    # গ্রিড আকারে ছবি দেখানোর জন্য কলাম তৈরি
    cols = st.columns(4) 
    col_idx = 0

    for index, row in df.iterrows():
        student_id = str(row['student_id'])
        student_name = row['name']
        
        # স্টুডেন্টের নির্দিষ্ট ফোল্ডার (dataset/student_id)
        student_dir = dataset_path / student_id
        
        if student_dir.exists() and student_dir.is_dir():
            # ফোল্ডারের ভেতর থেকে যেকোনো একটি ছবি নেওয়া
            images = [img for img in os.listdir(student_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
            
            if images:
                img_path = student_dir / images[0] # প্রথম ছবিটি দেখাচ্ছে
                
                with cols[col_idx]:
                    st.image(str(img_path), caption=f"{student_id} - {student_name}", use_container_width=True)
                
                col_idx = (col_idx + 1) % 4
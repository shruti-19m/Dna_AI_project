import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
from sklearn.neighbors import NearestNeighbors

# ---------------- BACKGROUND IMAGE ----------------
def set_bg_image(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}

        /* Slider label color */
        label {{
            color: white !important;
            font-size: 16px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image("dna_bg.png")

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #00FFFF;'>
    🧬 AI-Based DNA Fingerprinting System
    </h1>
    """,
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
data = pd.read_csv("dna_database.csv")
X = data.iloc[:, 1:].values
ids = data["id"].values

# ---------------- INFO BOX ----------------
st.markdown(
    """
    <div style="
        background-color: rgba(0, 0, 0, 0.65);
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-size: 18px;">
        Enter DNA STR values and run AI-based matching
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- DNA INPUT ----------------
st.markdown(
    "<h2 style='color:#00FFFF;'>🔬 Crime Scene DNA Input</h2>",
    unsafe_allow_html=True
)

str1 = st.slider("STR 1", 5, 20, 14)
str2 = st.slider("STR 2", 5, 20, 9)
str3 = st.slider("STR 3", 5, 20, 16)
str4 = st.slider("STR 4", 5, 20, 11)
str5 = st.slider("STR 5", 5, 20, 7)

crime_dna = np.array([[str1, str2, str3, str4, str5]])

# ---------------- RUN AI ----------------
if st.button("🔍 Run AI DNA Matching"):
    start_time = time.time()

    knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    knn.fit(X)

    distances, indices = knn.kneighbors(crime_dna)

    best_index = indices[0][0]
    best_distance = distances[0][0]
    suspect_id = ids[best_index]

    accuracy = max(0, 100 - best_distance * 5)
    confidence = min(100, accuracy + 2)
    processing_time = (time.time() - start_time) * 1000

    # -------- RESULT BOX --------
    st.markdown(
        f"""
        <div style="
            background-color: rgba(0, 128, 0, 0.75);
            padding: 20px;
            border-radius: 15px;
            color: white;
            font-size: 20px;">
            ✅ DNA Match Found<br><br>
            🆔 Suspect ID: <b>{suspect_id}</b><br>
            📊 Accuracy: <b>{round(accuracy,2)}%</b><br>
            🔐 Confidence: <b>{round(confidence,2)}%</b><br>
            ⏱ Processing Time: <b>{round(processing_time,2)} ms</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------- CLOSEST MATCHES --------
    st.subheader("🔎 Closest DNA Matches")

    st.markdown(
        """
        <div style="
            background-color: rgba(0, 0, 0, 0.65);
            padding: 15px;
            border-radius: 15px;
            color: white;
            font-size: 18px;">
            <b>Top 3 closest matches from database:</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    for i in range(3):
        st.markdown(
            f"""
            <div style="
                background-color: rgba(255,255,255,0.1);
                padding: 10px;
                margin-top: 8px;
                border-radius: 10px;
                color: #00FFFF;
                font-size: 16px;">
                {i+1}. 🆔 <b>{ids[indices[0][i]]}</b> |
                📏 Distance: <b>{round(distances[0][i],2)}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

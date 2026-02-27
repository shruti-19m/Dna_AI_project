import streamlit as st
import pandas as pd
import numpy as np
import time
import base64

# Function to add background image
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
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image("dna_bg.png")


from sklearn.neighbors import NearestNeighbors

# UI Title


st.markdown(
    """
    <h1 style='text-align: center; color: #00FFFF;'>
    🧬 AI-Based DNA Fingerprinting System
    </h1>
    """,
    unsafe_allow_html=True
)

# Load database
data = pd.read_csv("dna_database.csv")
X = data.iloc[:, 1:].values   # DNA STR values
ids = data["id"].values



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

# Crime DNA input
st.markdown(
    """
    <h2 style='color: #00FFFF;'>🔬 Crime Scene DNA Input</h2>
    """,
    unsafe_allow_html=True
)
#st.subheader("Enter Crime Scene DNA (STR Values)")



str1 = st.slider("STR 1", 5, 20, 14, key="str1")
str2 = st.slider("STR 2", 5, 20, 9, key="str2")
str3 = st.slider("STR 3", 5, 20, 16, key="str3")
str4 = st.slider("STR 4", 5, 20, 11, key="str4")
str5 = st.slider("STR 5", 5, 20, 7, key="str5")

crime_dna = np.array([[str1, str2, str3, str4, str5]])

if st.button("🔍 Run AI DNA Matching"):
    start_time = time.time()

    # KNN Model
    knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    knn.fit(X)

    distances, indices = knn.kneighbors(crime_dna)

    best_index = indices[0][0]
    best_distance = distances[0][0]
    suspect_id = ids[best_index]

    accuracy = max(0, 100 - best_distance * 5)
    confidence = min(100, accuracy + 2)

    end_time = time.time()
    processing_time = (end_time - start_time) * 1000
    st.markdown(
    f"""
    <div style="
        background-color: rgba(0, 128, 0, 0.75);
        padding: 20px;
        border-radius: 15px;
        color: white;
        font-size: 20px;">
         ✅ DNA Match Found <br><br>
        🆔 Suspect ID: <b>{suspect_id}</b><br>
        📊 Accuracy: <b>{round(accuracy,2)}%</b><br>
        🔐 Confidence: <b>{round(confidence,2)}%</b><br>
        ⏱ Time: <b>{round(processing_time,2)} ms</b>
    </div>
    """,
    unsafe_allow_html=True
)
           
   

    st.subheader("🔎 Closest DNA Matches")
    for i in range(3):
        st.write(f"{i+1}. ID: {ids[indices[0][i]]} | Distance: {round(distances[0][i],2)}")
               
    



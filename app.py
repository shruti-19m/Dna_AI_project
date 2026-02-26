import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

# UI Title
st.title("🧬 Advanced AI-Based DNA Fingerprinting System")

# Load database
data = pd.read_csv("dna_database.csv")
X = data.iloc[:, 1:].values   # DNA STR values
ids = data["id"].values

# Crime DNA input
st.subheader("Enter Crime Scene DNA (STR Values)")
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

    st.success("✅ DNA Match Found")
    st.write("🆔 Suspect ID:", suspect_id)
    st.write("📊 Match Accuracy:", round(accuracy, 2), "%")
    st.write("🔐 Confidence Score:", round(confidence, 2), "%")
    st.write("⏱ Processing Time:", round(processing_time, 2), "ms")

    st.subheader("🔎 Closest DNA Matches")
    for i in range(3):
        st.write(f"{i+1}. ID: {ids[indices[0][i]]} | Distance: {round(distances[0][i],2)}")

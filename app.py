import streamlit as st
import pandas as pd
import pickle

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.title("Automatic Clustering Algorithm Comparison")

# Load preprocessor (optional)
# preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Select numeric columns
    X = data.select_dtypes(include=['int64', 'float64'])

    if X.shape[1] < 2:
        st.error("Dataset must contain at least 2 numeric columns")
        st.stop()

    if st.button("Run Algorithms"):

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results = {}

        # KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels_k = kmeans.fit_predict(X_scaled)
        results["KMeans"] = silhouette_score(X_scaled, labels_k)

        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels_d = dbscan.fit_predict(X_scaled)

        if len(set(labels_d)) > 1:
            results["DBSCAN"] = silhouette_score(X_scaled, labels_d)
        else:
            results["DBSCAN"] = -1

        # Hierarchical
        hc = AgglomerativeClustering(n_clusters=3)
        labels_h = hc.fit_predict(X_scaled)
        results["Hierarchical"] = silhouette_score(X_scaled, labels_h)

        best_algo = max(results, key=results.get)

        st.subheader("Algorithm Scores")
        st.write(results)

        st.success(f"Best Algorithm: {best_algo}")
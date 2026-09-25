import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


# Page configuration
st.set_page_config(
    page_title="Credit Card DBSCAN",
    page_icon="💳",
    layout="wide"
)


# Title
st.title("💳 Credit Card Customers - DBSCAN Clustering")

st.write(
    "DBSCAN clustering analysis of Credit Card Customer data."
)


# Upload dataset
uploaded_file = st.file_uploader(
    "Upload CC GENERAL.csv",
    type=["csv"]
)


if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())


    # Features
    features = [
        "BALANCE",
        "PURCHASES",
        "CASH_ADVANCE",
        "CREDIT_LIMIT",
        "PAYMENTS"
    ]

    X = df[features]


    # Handle missing values
    X = X.fillna(X.median())


    # Scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)


    # Sidebar
    st.sidebar.header("DBSCAN Parameters")


    eps = st.sidebar.slider(
        "EPS",
        min_value=0.1,
        max_value=3.0,
        value=0.5,
        step=0.1
    )


    min_samples = st.sidebar.slider(
        "Minimum Samples",
        min_value=2,
        max_value=20,
        value=5,
        step=1
    )


    # DBSCAN
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )


    labels = dbscan.fit_predict(
        X_scaled
    )


    # Add cluster labels
    df["Cluster"] = labels


    # Number of clusters
    n_clusters = len(set(labels)) - (
        1 if -1 in labels else 0
    )


    # Noise
    noise_points = list(labels).count(-1)


    # Silhouette Score
    cluster_labels = set(labels) - {-1}


    if len(cluster_labels) >= 2:

        score = silhouette_score(
            X_scaled,
            labels
        )

    else:

        score = None


    # Results
    st.subheader("DBSCAN Results")


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Number of Clusters",
        n_clusters
    )


    col2.metric(
        "Noise Points",
        noise_points
    )


    if score is not None:

        col3.metric(
            "Silhouette Score",
            round(score, 4)
        )

    else:

        col3.metric(
            "Silhouette Score",
            "N/A"
        )


    # Cluster Distribution
    st.subheader("Cluster Distribution")


    st.write(
        df["Cluster"].value_counts()
    )


    # Visualization
    st.subheader("DBSCAN Visualization")


    fig, ax = plt.subplots(
        figsize=(10, 6)
    )


    scatter = ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=labels,
        cmap="viridis",
        s=30
    )


    ax.set_xlabel(
        "BALANCE"
    )


    ax.set_ylabel(
        "PURCHASES"
    )


    ax.set_title(
        "DBSCAN Clustering - Credit Card Customers"
    )


    plt.colorbar(
        scatter,
        ax=ax,
        label="Cluster"
    )


    st.pyplot(fig)


    # Dataset with clusters
    st.subheader(
        "Dataset with Cluster Labels"
    )


    st.dataframe(df)


    # Download
    csv_data = df.to_csv(
        index=False
    )


    st.download_button(
        label="⬇️ Download Clustered Dataset",
        data=csv_data,
        file_name="Credit_Card_DBSCAN_Clustered.csv",
        mime="text/csv"
    )
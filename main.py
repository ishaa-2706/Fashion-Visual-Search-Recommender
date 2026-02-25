import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Fashion Visual Search", layout="wide")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    styles_df = pd.read_csv("styles_cleaned.csv")
    products_df = pd.read_csv("products.csv")

    with open("embeddings.pkl", "rb") as f:
        features = np.array(pickle.load(f))

    with open("filenames.pkl", "rb") as f:
        filenames = pickle.load(f)

    assert len(features) == len(filenames), "Embeddings & filenames length mismatch!"

    return styles_df, products_df, features, filenames


# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    return model


# ---------------- Feature Extraction ----------------
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)

    features = model.predict(img_arr, verbose=0).flatten()
    features = features / norm(features)
    return features


# ---------------- Infer Category ----------------
def infer_category(query_vec, features, filenames, styles_df, top_k=20):
    nn = NearestNeighbors(n_neighbors=min(top_k, len(features)), metric="euclidean")
    nn.fit(features)
    distances, indices = nn.kneighbors([query_vec])

    ids = []
    for idx in indices[0]:
        try:
            pid = int(os.path.basename(filenames[idx]).split(".")[0])
            ids.append(pid)
        except:
            pass

    matched = styles_df[styles_df["id"].isin(ids)]
    if len(matched) == 0:
        return None, None

    category = matched["subCategory"].mode()[0]
    confidence = float(distances[0][1])  # distance of closest neighbor
    return category, confidence


# ---------------- Recommend Inside Category ----------------
def recommend_in_category(query_vec, category, features, filenames, styles_df, k=5):
    category_ids = styles_df[styles_df["subCategory"].str.lower() == category.lower()]["id"].tolist()

    idx_map = []
    for i, f in enumerate(filenames):
        try:
            pid = int(os.path.basename(f).split(".")[0])
            if pid in category_ids:
                idx_map.append(i)
        except:
            pass

    if len(idx_map) == 0:
        return []

    category_features = features[idx_map]

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(category_features)), metric="euclidean")
    nn.fit(category_features)

    distances, indices = nn.kneighbors([query_vec])

    results = []
    for i in indices[0][1:]:
        results.append(filenames[idx_map[i]])

    return results


# ---------------- UI ----------------
def main():
    st.title("🧥 Fashion Visual Search & Recommender")

    styles_df, products_df, features, filenames = load_data()
    model = load_model()

    uploaded_file = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(file_path, caption="Uploaded Image", width=250)

        if st.button("🔍 Find Similar Products"):
            with st.spinner("Analyzing image..."):
                query_vec = extract_features(file_path, model)

                category, confidence = infer_category(query_vec, features, filenames, styles_df)

                if category is None or confidence > 1.3:
                    st.error("⚠️ This image does not belong to fashion dataset.")

                    st.markdown("### 🔗 Explore Online:")
                    st.markdown("[🛒 Flipkart Fashion](https://www.flipkart.com/clothing-and-accessories/pr?sid=clo)")
                    st.markdown("[🛒 Amazon Fashion](https://www.amazon.in/fashion)")
                    return

                st.success(f"🎯 Detected product type: {category}")

                results = recommend_in_category(query_vec, category, features, filenames, styles_df)

                if len(results) == 0:
                    st.warning("No similar products found in database.")

                    st.markdown("### 🔗 Explore Online:")
                    st.markdown(f"[🛒 Flipkart {category}](https://www.flipkart.com/search?q={category})")
                    st.markdown(f"[🛒 Amazon {category}](https://www.amazon.in/s?k={category})")
                    return

                st.subheader("✨ Visually Similar Products")

                cols = st.columns(5)
                for i, img_path in enumerate(results):
                    pid = int(os.path.basename(img_path).split(".")[0])
                    row = products_df[products_df["id"] == pid].iloc[0]

                    with cols[i % 5]:
                        st.image(img_path, use_column_width=True)
                        st.caption(row["productDisplayName"])
                        st.markdown(f"[🛒 Flipkart]({row['flipkart_url']}) | [🛒 Amazon]({row['amazon_url']})")

    st.markdown("---")
    st.caption("Built with CNN embeddings + Nearest Neighbors")


if __name__ == "__main__":
    main()
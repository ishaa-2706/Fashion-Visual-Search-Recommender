import pandas as pd
import urllib.parse

df = pd.read_csv("styles.csv", engine="python", on_bad_lines="skip")
df.to_csv("styles_cleaned.csv", index=False)

def make_query(name, category):
    q = f"{name} {category}"
    return urllib.parse.quote_plus(str(q))

products = pd.DataFrame({
    "id": df["id"],
    "image_path": "images/" + df["id"].astype(str) + ".jpg",
    "gender": df["gender"].astype(str).str.lower(),
    "subCategory": df["subCategory"].astype(str).str.lower(),
    "articleType": df["articleType"].astype(str).str.lower(),
    "baseColour": df["baseColour"].astype(str).str.lower(),
    "productDisplayName": df["productDisplayName"],

    # 🔗 Real shopping links (search-based)
    "flipkart_url": df.apply(
        lambda x: f"https://www.flipkart.com/search?q={make_query(x['productDisplayName'], x['articleType'])}",
        axis=1
    ),
    "amazon_url": df.apply(
        lambda x: f"https://www.amazon.in/s?k={make_query(x['productDisplayName'], x['articleType'])}",
        axis=1
    ),
})

products.to_csv("products.csv", index=False)
print("products.csv created successfully!")
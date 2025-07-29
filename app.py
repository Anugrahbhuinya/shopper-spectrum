import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu

# --- Load Models ---
scaler = joblib.load("scaler_rfm.pkl")
kmeans = joblib.load("kmeans_rfm.pkl")

# --- Load and Clean Data ---
df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')
df = df.dropna(subset=['CustomerID', 'Description'])
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['CustomerID'] = df['CustomerID'].astype(int)
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# --- Build Product Similarity Matrix ---
pivot = df.pivot_table(index='StockCode', columns='CustomerID', values='Quantity', aggfunc='sum', fill_value=0)
product_sim = cosine_similarity(pivot)
product_sim_df = pd.DataFrame(product_sim, index=pivot.index, columns=pivot.index)
product_names = df.groupby('StockCode')['Description'].agg(lambda x: x.mode()[0])

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Shopper Spectrum", layout="wide")

# --- Sidebar Menu using streamlit-option-menu ---
with st.sidebar:
    selected = option_menu(
        menu_title=" Main Menu",
        options=["Home", "Clustering", "Recommendation"],
        icons=["house", "people", "cart"],
        menu_icon="shop",  # Icon next to "Main Menu"
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#0f0e0e"},
            "icon": {"color": "#ff4b4b", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#ff4b4b", "color": "white"},
        }
    )

# --- Home Page ---
if selected == "Home":
    st.title(" Welcome to Shopper Spectrum")
    st.write("Use the sidebar to navigate:")
    st.markdown("""
    -  **Clustering Module:** RFM-based customer segmentation  
    -  **Recommendation Module:** Get similar product suggestions  
    """)

# --- Clustering Page ---
elif selected == "Clustering":
    st.title(" Customer Segmentation (RFM + KMeans)")
    recency = st.number_input("Recency (days)", min_value=0, value=30)
    frequency = st.number_input("Frequency (purchases)", min_value=0, value=10)
    monetary = st.number_input("Monetary Value (£)", min_value=0.0, value=1000.0)

    if st.button("Predict Cluster"):
        scaled = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(scaled)[0]

        # Update this mapping as per your cluster evaluation
        cluster_map = {
            2: 'High-Value',
            3: 'Regular',
            0: 'Occasional',
            1: 'At-Risk'
        }

        segment = cluster_map.get(cluster, "Unknown")
        st.success(f" Predicted Segment: **{segment}**")

# --- Recommendation Page ---
elif selected == "Recommendation":
    st.title(" Product Recommender")
    product_input = st.text_input("Enter Product Name", placeholder="e.g. GREEN VINTAGE SPOT BEAKER")

    if st.button("Recommend"):
        matches = product_names[product_names.str.contains(product_input, case=False, na=False)]

        if matches.empty:
            st.error(" Product not found. Try another.")
        else:
            stock_code = matches.index[0]
            st.success(f" Found: {product_names[stock_code]}")
            similar = product_sim_df[stock_code].sort_values(ascending=False)[1:6].index
            recommendations = product_names[similar].tolist()

            st.subheader("Recommended Products:")
            for i, prod in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {prod}")

# --- Footer ---
st.markdown("---")
st.caption(" Built with  for the Shopper Spectrum Capstone Project")

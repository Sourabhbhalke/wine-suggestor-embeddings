import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import os

# Set page configuration first (must be at the top)
st.set_page_config(page_title="Wine Explorer App", layout="wide")

# Load and display the wine dataset
@st.cache_data
def load_data():
    # Replace with your dataset path or logic to read data
    df = pd.read_csv('top_rated_wines.csv')  # Ensure this file exists in the project directory
    return df

# Initialize SentenceTransformer encoder and Qdrant client
@st.cache_resource
def load_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def create_qdrant_client():
    return QdrantClient(":memory:")

# Initialize Qdrant and load embeddings
encoder = load_encoder()
qdrant = create_qdrant_client()

# Pre-run embeddings generation and upload to Qdrant
@st.cache_resource
def generate_embeddings():
    data = load_data()
    wine_data = data.to_dict('records')
    points = [
        models.PointStruct(
            id=idx,
            vector=encoder.encode(wine["notes"]).tolist(),
            payload=wine,
        )
        for idx, wine in enumerate(wine_data)
    ]
    qdrant.recreate_collection(
        collection_name="top_wines",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        )
    )
    qdrant.upload_points(collection_name="top_wines", points=points)
    return data  # Return data after embeddings are generated and uploaded

# Call the function to generate and cache embeddings
data = generate_embeddings()

# Page selection for filters or AI-powered query
page_option = st.sidebar.radio("Select Page Option", ["Choose with Filters", "Choose by Asking AI"])

# 1. Choose with Filters Page
if page_option == "Choose with Filters":
    st.title("Wine Explorer - Filtered Selection")
    st.sidebar.header("Filter by:")

    # Sidebar filters
    region = st.sidebar.selectbox("Select region:", data['region'].unique())
    variety = st.sidebar.selectbox("Select variety:", data['variety'].unique())

    filtered_data = data[(data['region'] == region) & (data['variety'] == variety)]
    st.write("### Filtered Wines", filtered_data)

    # Visualization (example of a rating distribution bar chart)
    st.write("### Rating Distribution")
    rating_counts = data['rating'].value_counts()
    st.bar_chart(rating_counts)

    # Improve the layout (e.g., using columns for better UI)
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Wine Variety Distribution")
        st.bar_chart(data['variety'].value_counts())

    with col2:
        st.write("### Wine Region Distribution")
        st.bar_chart(data['region'].value_counts())

# 2. Choose by Asking AI Page
if page_option == "Choose by Asking AI":
    st.title("Wine Explorer - AI-Powered Suggestions")

    # Query using embeddings
    user_query = st.text_input("Enter a query (e.g., 'Suggest me a Malbec wine from Argentina'):")

    if user_query:
        # Convert user query to a vector using the encoder
        query_vector = encoder.encode(user_query).tolist()

        # Search for similar wines based on the query vector
        hits = qdrant.search(
            collection_name="top_wines",
            query_vector=query_vector,
            limit=3,
        )

        # Display search results
        st.write("### Search Results")
        for hit in hits:
            st.write(hit.payload)  # Display the payload (wine details)

        # AI-powered suggestions (mock example)
        st.write("### AI-Powered Suggestions")
        suggestions = [f"AI Suggestion {i+1} for '{user_query}'" for i in range(3)]
        for suggestion in suggestions:
            st.write(suggestion)

    # Optional: Add a reset button for queries
    if st.button("Reset AI Query"):
        st.text_input("Enter a query (e.g., 'Suggest me a Malbec wine from Argentina'):", value="", key="reset_input")

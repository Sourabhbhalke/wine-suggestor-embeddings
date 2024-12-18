# Wine Explorer App - Implementing the Retrieval Augmented Generation (RAG) Pattern
![image](https://github.com/user-attachments/assets/0f7a48ff-084c-48d9-bc99-aa21b4bdcb89)

This app implements the **Retrieval Augmented Generation (RAG)** pattern using your own data. In this case, the app is designed to explore a wine dataset and use natural language queries to recommend wines based on user preferences, powered by embeddings and a vector database (Qdrant).

## Learning Objectives

- Implement the **RAG** pattern with your own data
- Apply your own data to solve a problem using the RAG pattern
- Understand how to leverage a **Large Language Model (LLM)** and a **vector database** like **Qdrant** for useful responses

## Features

- **Filtered Selection**: Users can filter wines based on region and variety, viewing relevant recommendations and visualizations.
- **AI-Powered Suggestions**: Users can input natural language queries, and the app will use **Sentence Transformers** to generate embeddings and search for similar wines from the dataset stored in **Qdrant**.
- **Pre-run Embeddings**: The app generates wine embeddings in advance, ensuring that the AI-powered query results are ready without any delay.

## Technologies Used

- **Streamlit**: For building the web app interface
- **Pandas**: For data manipulation and handling the wine dataset
- **Sentence-Transformers**: To convert wine notes into embeddings for similarity search
- **Qdrant**: A vector database for storing and searching embeddings
- **OpenAI API (optional)**: To integrate an LLM (if needed for further queries)

## Steps

1. **Prepare Your Data**: 
   - Format your wine dataset as a list of dictionaries for easier ingestion into the vector database.
   - The dataset should contain at least `region`, `variety`, `rating`, and `notes` (description or tasting notes) for each wine.

2. **Generate Embeddings**:
   - The app uses **Sentence-Transformers** to convert the `notes` of each wine into embeddings, which are uploaded to **Qdrant** for similarity search.

3. **Run the App**:
   - The app is split into two modes:
     - **Choose with Filters**: Filter wines by region and variety.
     - **Choose by Asking AI**: Query wines by natural language.
   - Once embeddings are generated and uploaded, users can interact with the app by entering queries like "Suggest me a Malbec wine from Argentina" to receive AI-powered recommendations.

4. **Integrate with LLM** (optional):
   - If desired, you can extend the functionality by connecting the app to an OpenAI endpoint, allowing for more advanced, context-aware AI responses.

## Concepts Covered

- **Retrieval Augmented Generation (RAG)**: Using a vector database (Qdrant) to retrieve relevant data before generating responses with an LLM.
- **Large Language Models (LLMs)**: How to use LLMs to process and generate human-like responses based on user input.
- **Vector Databases**: Using **Qdrant** for storing and searching high-dimensional data (embeddings).
- **Sentence Transformers**: Converting text into embeddings for similarity comparison.
- **OpenAI Python API**: Connecting to an LLM and generating advanced responses (optional step).

## How It Works

1. **Data Processing**:
   - The app loads the wine dataset and processes the `notes` of each wine using the **Sentence-Transformer** model to create embeddings.
   - These embeddings are uploaded to **Qdrant**, allowing for fast retrieval of similar wines based on a user's query.

2. **User Interaction**:
   - **Filtered Selection**: Users can filter wines by region and variety.
   - **AI-Powered Querying**: Users can type in natural language queries, which are converted into embeddings and compared with the stored wine embeddings in Qdrant to find similar wines.

3. **Visualizations**:
   - The app provides charts to visualize the distribution of wine varieties and regions, and a bar chart for ratings.

4. **AI-Powered Suggestions**:
   - When a user inputs a query, the app retrieves the most similar wines using the Qdrant vector search and displays them with their relevant details.

## Running the App

1. Install the required dependencies:

  
   pip install streamlit pandas sentence-transformers qdrant-client
   
2. Run the app:


streamlit run wines-app.py
![image](https://github.com/user-attachments/assets/8b036628-50da-4ffb-b9cc-68c8a91cf23c)

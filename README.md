
# 📚 Book Recommendation System

**Live Demo** 👉 [Hugging Face Space](https://huggingface.co/spaces/Arshdeep004/book)

This is a smart book recommendation system powered by Natural Language Processing (NLP) and Large Language Models (LLMs). It helps users find books based on themes, genres, emotions, and user queries through a simple and interactive Gradio web interface.

---

## 🔍 Features

### 1. 📊 Exploratory Data Analysis (EDA)
- Analyzed the dataset to understand book titles, genres, authors, and ratings.
- Identified patterns and trends across the data using visualizations.

### 2. 🧼 Text Data Cleaning
- Cleaned book descriptions by removing punctuation, stopwords, and other noise.
- Tokenized and normalized text for further processing.

### 3. 🔎 Semantic Search (Vector Embeddings)
- Used `sentence-transformers` to generate embeddings for book descriptions.
- Created a vector database using ChromaDB and LangChain.
- Enabled natural language search (e.g., "a book about a person seeking revenge") to retrieve relevant books based on semantic similarity.

### 4. 🏷️ Zero-Shot Text Classification
- Applied zero-shot classification using Hugging Face’s `pipeline` to categorize books as **Fiction** or **Non-Fiction**.
- No labeled dataset needed — the model infers labels from context.

### 5. 🎭 Sentiment and Emotion Analysis
- Used LLMs to extract emotions and tone (e.g., joyful, suspenseful, sad) from book descriptions.
- Users can filter and sort books by emotional tone.

### 6. 🌐 Gradio Web Application
- Developed an interactive web interface using Gradio.
- Users can:
  - Input natural language queries
  - Filter books by fiction/non-fiction
  - Sort by emotion/tone
- Deployed live on Hugging Face Spaces.

---

## 🧠 Technologies Used

| Feature                     | Tools/Libraries                          |
|-----------------------------|------------------------------------------|
| EDA & Cleaning              | pandas, seaborn, regex                   |
| Embeddings & Semantic Search| sentence-transformers, ChromaDB, LangChain |
| Classification              | Hugging Face Transformers (zero-shot)   |
| Sentiment & Emotion         | LLMs via Hugging Face API               |
| Web Interface               | Gradio                                   |
| Deployment                  | Hugging Face Spaces                      |

---


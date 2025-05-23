import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr

load_dotenv()

# Load and preprocess data
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "not_found.jpg",
    books["large_thumbnail"],
)

# Load document embeddings
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding_model)

# Recommendation logic
def retrieve_semantic_recommendations(query, category=None, tone=None, initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs

# Gradio recommendation interface
def recommend_books(query, category, tone):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"**{row['title']}** by _{authors_str}_\n\n{truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Gradio app with improved UX
with gr.Blocks(theme=gr.themes.Base(), css="""
#main-container { max-width: 1200px; margin: auto; }
.gallery-item img { border-radius: 12px; transition: transform 0.3s ease-in-out; }
.gallery-item:hover img { transform: scale(1.05); }
""") as dashboard:

    with gr.Column(elem_id="main-container"):
        gr.Markdown("# 📚 Semantic Book Recommender")
        gr.Markdown("Describe a book you're in the mood for, and get personalized recommendations based on emotion and genre.")

        with gr.Row():
            user_query = gr.Textbox(
                label="🔍 What kind of book are you looking for?",
                placeholder="e.g., A magical journey of friendship and discovery...",
                lines=2,
            )
        with gr.Row():
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="📖 Filter by genre:",
                value="All"
            )
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="🎭 Desired emotional tone:",
                value="All"
            )

        submit_button = gr.Button("✨ Recommend Books", size="lg")

        gr.Markdown("## 🔎 Top Matches")
        output = gr.Gallery(
            label="Recommended Books",
            show_label=False,
            columns=4,
            object_fit="cover",
            height="auto",
            elem_classes=["gallery-item"]
        )

        submit_button.click(
            fn=recommend_books,
            inputs=[user_query, category_dropdown, tone_dropdown],
            outputs=output
        )

if __name__ == "__main__":
    dashboard.launch()

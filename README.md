
# 📚 Graph-RAG: Smarter Question Answering with Graphs + AI

Ever wished your documents could talk back? **Graph-RAG** does just that.

This project combines the power of **knowledge graphs** and **AI embeddings** to help you ask meaningful questions about any PDF — and get smart, contextual answers in return.

## ✨ What It Does

- 📄 Upload a PDF and break it down into chunks.
- 🧠 Understands each sentence using sentence embeddings.
- 🌐 Builds a **knowledge graph** in Neo4j to map relationships.
- 🔍 Uses **both graph queries and vector search** to find the best answer.
- 💬 Responds using a Language Model that understands your query context.

## 🗂️ What's Inside

```
Graph-RAG/
├── app.py               → The user interface (built with Streamlit)
├── main.py              → Runs the full pipeline: parse → graph → embed
├── embedding_model.py   → Turns text into meaningful vectors
├── requirements.txt     → All the libraries you'll need
├── template/index.html  → UI template for the web app
└── README.md            → You’re reading it!
```

## 🚀 How to Run It

1. **Clone this project**
   ```bash
   git clone https://github.com/yourusername/graph-rag.git
   cd graph-rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Make sure Neo4j is running**
   - You can use [Neo4j Desktop](https://neo4j.com/download/)
   - Set your DB credentials in the script or environment

4. **Launch the app**
   ```bash
   streamlit run app.py
   ```

5. **Start chatting with your PDF!**
   - Upload a document
   - Ask: _“What are the key points in section 3?”_
   - Get structured + semantic answers

## 🧠 Built With

- `LangChain` – manages how the AI responds
- `FAISS` – fast vector search engine
- `Neo4j` – builds a relationship graph from your document
- `pdfplumber` – extracts clean text from PDFs
- `sentence-transformers` – makes text machine-understandable

## 💡 Why Use Graph + Vector Together?

Think of it like this:
- **Vectors** give you what _sounds similar_.
- **Graphs** give you what’s _connected logically_.

Using both gives you answers that are **relevant and accurate**.

## 🛠 Future Plans

- Add document summaries
- Keep chat memory so it feels like a real convo
- Package everything into Docker for easy deployment


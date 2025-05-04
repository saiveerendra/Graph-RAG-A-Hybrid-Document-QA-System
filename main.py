from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from embedding_model import embedding_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.pydantic_v1 import BaseModel, Field

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

groq_api_key = "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"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

os.environ["NEO4J_URI"] = "neo4j+s://39948ffb.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "rKfg6XGcpE_jX_jVQ8IAQD9TKDRMkEBg3-ENkfZklP8"
graph = Neo4jGraph()

vector_index = None  # cache


def load_and_split_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pdf_filename = os.path.basename(pdf_path)
    text_splitter = TokenTextSplitter(chunk_size=512,
                                   chunk_overlap=50,
                                  allowed_special=set(["\n\n", "\n", ".", " "])) # Changed to set
# Create a list of Document objects from the fitz.Page objects
    documents = [Document(page_content=page.get_text(), metadata={"pdf_name": pdf_filename}) for page in doc]
    return text_splitter.split_documents(documents)
   


def extract_graph(documents):
    graph_prompt = ChatPromptTemplate.from_template("""
Extract a graph from the text. Define:
- Entities with ID, type, and properties
- Relationships with source, target, type, and properties

Example:
Text: "Tom Hanks acted in Toy Story in 1995."

NODE:
id: tom_hanks
type: Person
properties: {{"name": "Tom Hanks"}}

NODE:
id: toy_story
type: Movie
properties: {{"title": "Toy Story", "release_year": "1995"}}

EDGE:
type: ACTED_IN
source: tom_hanks
target: toy_story
properties: {{"year": "1995"}}

Now extract from:
{input}
""")
    llm_transformer = LLMGraphTransformer(llm=llm, prompt=graph_prompt)
    graph_docs = llm_transformer.convert_to_graph_documents(documents)
    graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)

from neo4j import GraphDatabase

def get_graph_data(cypher: str):
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    with driver.session() as session:
        result = session.run(cypher)
        nodes = {}
        links = []

        for record in result:
            for key in record.keys():
                graph = record[key]
                for node in graph.nodes:
                    nodes[node.id] = {
                        "id": node.id,
                        "label": node.get("id", ""),
                        "type": node.labels.pop() if node.labels else "Node",
                        "properties": dict(node.items())
                    }
                for rel in graph.relationships:
                    links.append({
                        "source": rel.start_node.id,
                        "target": rel.end_node.id,
                        "type": rel.type,
                        "properties": dict(rel.items())
                    })

    return {
        "nodes": list(nodes.values()),
        "links": links
    }

def embed_documents():
    global vector_index
    if vector_index is None:
        vector_index = Neo4jVector.from_existing_graph(
            embedding=embedding_model,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
    return vector_index


class Entities(BaseModel):
    names: list[str] = Field(..., description="Person or organization entities from text.")


entity_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are extracting organization and person entities from the text."),
    ("human", "Extract from: {question}"),
])
entity_chain = entity_prompt | llm.with_structured_output(Entities)


def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    return " AND ".join(f"{word}~2" for word in words)


def structured_retriever(question: str) -> str:
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        query = generate_full_text_query(entity)
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
               YIELD node,score
               CALL {
                 WITH node
                 MATCH (node)-[r]->(neighbor) RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                 UNION ALL
                 WITH node
                 MATCH (node)<-[r]-(neighbor) RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
               }
               RETURN output LIMIT 50""",
            {"query": query},
        )
        result += "\n".join([el["output"] for el in response]) + "\n"
    return result


def retriever(question: str) -> str:
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    vector_index = embed_documents()
    results = vector_index.similarity_search(question)
    unstructured_data = []
    pdf_list = []
    seen_pdf_names = set()  # Keep track of seen pdf_names
    for doc in results:
        pdf_name = doc.metadata.get("pdf_name")
        if pdf_name not in seen_pdf_names:  # Check if pdf_name is already seen
            pdf_list.append(pdf_name)  # Append if not seen before
            seen_pdf_names.add(pdf_name)  # Add to seen_pdf_names
        content = doc.page_content.strip()
        unstructured_data.append(f"📄 From: {pdf_name}\n{content}")

    return {
        "context": f"""Structured data:\n{structured_data}\n\nUnstructured data:\n{"#Document ".join(unstructured_data)}""",
        "sources": list(pdf_list)
    }


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
Given the chat and a follow-up question, rewrite it as a standalone question.

Chat History:
{chat_history}
Follow Up: {question}
Standalone Question:
""")


def _format_chat_history(chat: list[tuple[str, str]]) -> list:
    messages = []
    for human, ai in chat:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    return messages


_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))),
        RunnablePassthrough.assign(chat_history=lambda x: _format_chat_history(x["chat_history"]))
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    RunnableLambda(lambda x: x["question"]),
)

final_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the **given context only** to answer the user's question. 
If the context is brief, expand slightly (2–3 lines) while staying accurate. Ensure the answer is clear, well-structured, and uses natural language. and we have Source: ['pdf_name.pdf'] like these so extract names of pdf and add it in last of output:


Context:
{context}

Question:
{question}

Instructions:
- Be concise, but informative.
- Do not make up any facts.
- Format the answer neatly with paragraphs or bullet points if needed.
- At the end, extract and list all PDF filenames mentioned in `Source: [...]` as the sources.

Answer:

Sources:
List the PDF filenames used to answer the question (e.g., 'file1.pdf', 'report2023.pdf').
""")


def get_chain():
    return (
        RunnableParallel({
            "context": _search_query | retriever,
            "question": RunnablePassthrough()
        })
        | final_prompt
        | llm
        | StrOutputParser()
    )


qa_chain = get_chain()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['pdf_file']
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            documents = load_and_split_pdf(filepath)
            extract_graph(documents)
            embed_documents() 
            graph_data = get_graph_data("""CALL {
  MATCH (n)-[r]->(m)
  RETURN n, r, m LIMIT 100
}
""")
            return jsonify({'message': 'File uploaded and processed successfully', 'graph': graph_data}), 200
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Only PDFs are allowed.'}), 400


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    question = data['question']
    try:
        answer = qa_chain.invoke({"question": question})
        return jsonify({'answer': answer}), 200
    except Exception as e:
        return jsonify({'error': f'Error during query: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)

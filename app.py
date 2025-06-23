import os
from flask import Flask, request, render_template, jsonify
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings as LCEmbeddings
from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
VECTOR_STORE_PATH = "chroma_db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# IBM Watsonx credentials from environment variables
apikey = os.getenv("WATSONX_APIKEY")
url = os.getenv("WATSONX_URL")
# instance_id = os.getenv("WATSONX_INSTANCE_ID")
project_id = os.getenv("WATSONX_PROJECT_ID")

# Basic validation for credentials
if not all([apikey, url, project_id]):
    raise ValueError("Watsonx credentials not found in environment variables. Please check your .env file.")

# Type assertions after validation
assert apikey is not None
assert url is not None
# assert instance_id is not None
assert project_id is not None

credentials = {
    "apikey": apikey,
    "url": url,
    # "instance_id": instance_id,
}

# Watsonx LLM setup for IBM Public Cloud
llm = WatsonxLLM(
    model_id="ibm/granite-13b-instruct-v2",
    url=SecretStr(url),
    apikey=SecretStr(apikey),
    # instance_id=SecretStr(instance_id),
    project_id=project_id,
    params={"decoding_method": "greedy", "max_new_tokens": 300, "stop_sequences": ["<|endoftext|>"]}
)

# Custom embedding wrapper to match LangChain
class SlateEmbedding(LCEmbeddings):
    def __init__(self):
        self.embedding_model = Embeddings(
            model_id="ibm/slate-30m-english-rtrvr",
            credentials=credentials,
            project_id=project_id
        )

    def embed_documents(self, texts):
        return self.embedding_model.embed_documents(texts)

    def embed_query(self, text):
        return self.embedding_model.embed_documents([text])[0]

embedding = SlateEmbedding()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        file = request.files.get("textfile")
        question = request.form.get("question")

        if not question:
            return jsonify({"error": "Question is required."}), 400

        if not file:
            return jsonify({"error": "File is required."}), 400
        
        filename = file.filename
        if not filename:
            return jsonify({"error": "Invalid filename."}), 400
            
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Use a persistent directory for ChromaDB
        persist_directory = os.path.join(VECTOR_STORE_PATH, os.path.splitext(filename)[0])
        
        # Load and split text
        loader = TextLoader(filepath, encoding="utf-8")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = splitter.split_documents(documents)

        # Embedding and retrieval
        vectordb = Chroma.from_documents(
            documents=texts, 
            embedding=embedding, 
            persist_directory=persist_directory
        )
        vectordb.persist()
        
        retriever = vectordb.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

        result = qa.invoke({"query": question})
        
        # Cleanup the created vectordb to avoid using stale data, as we are creating it on every request for now
        # A more advanced implementation would cache this.
        vectordb.delete_collection()

        return jsonify({"answer": result["result"]})

    except Exception as e:
        app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500
# print("APIKEY:", os.getenv("WATSONX_APIKEY"))
# print("URL:", os.getenv("WATSONX_URL"))
# print("PROJECT_ID:", os.getenv("WATSONX_PROJECT_ID"))


if __name__ == "__main__":
    app.run(debug=True, port=5001)

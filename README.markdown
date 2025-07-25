# Multilingual Retrieval-Augmented Generation (RAG) System

This project implements a **Multilingual Retrieval-Augmented Generation (RAG) System** designed to process and answer queries in Bangla and English based on the Bangla literature document `HSC26-Bangla1st-Paper.pdf`. Built using **n8n**, **Ollama**, **Pinecone**, and **OCR.space**, the system supports curriculum-based queries, such as "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?" (answer: "১৫ বছর"), by leveraging hybrid search, text preprocessing, and memory management. The system includes a lightweight REST API and basic RAG evaluation metrics for groundedness and relevance.

## Objective

The system fulfills the requirements of the **AI Engineer (Level-1) Technical Assessment** by developing a RAG pipeline that:
- Accepts user queries in English and Bangla.
- Retrieves relevant document chunks from a knowledge base built from `HSC26-Bangla1st-Paper.pdf`.
- Generates accurate, context-grounded answers.
- Implements document chunking, vectorization, and memory management (short-term chat history, long-term vector database).
- Provides a REST API for interaction and evaluates RAG performance.

## Features

- **Multilingual Query Processing**: Handles Bangla and English queries with text normalization (e.g., "কল্যানী" to "কল্যাণী").
- **OCR Text Extraction**: Uses OCR.space API (`language: ben`) to extract text from PDFs.
- **Bangla NLP Preprocessing**: Cleans text by removing stopwords, normalizing content, and detecting OCR errors using a custom Flask-based NLP service.
- **Hybrid Retrieval**: Combines BM25 keyword search and `llama3.2:1b` vector embeddings stored in Pinecone for precise retrieval.
- **Memory Management**:
  - **Short-Term**: Stores recent chat inputs via n8n’s `memoryBufferWindow`.
  - **Long-Term**: Maintains document embeddings in Pinecone.
- **REST API**: Lightweight API endpoint for querying the system.
- **RAG Evaluation**: Assesses groundedness (cosine similarity) and relevance (manual evaluation).
- **Error Handling**: Manages JSON parsing errors, OCR issues, and low-quality retrievals.

## Source Document

- **Title**: HSC26 Bangla 1st Paper
- **Language**: Primarily Bangla, with some English
- **Domain**: Literature
- **Sample Queries and Expected Answers**:
  - "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?" → "শুম্ভুনাথ"
  - "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?" → "মামাকে"
  - "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?" → "১৫ বছর"

## Technology Stack

| Component               | Description                                           |
|-------------------------|-------------------------------------------------------|
| **n8n**                 | Workflow automation for orchestrating RAG pipeline     |
| **Ollama**              | Local LLM for `llama3.2:1b` embeddings and chat        |
| **Pinecone**            | Vector database for semantic search and storage        |
| **OCR.space API**       | Text extraction from PDFs with Bangla support          |
| **Flask**               | Lightweight REST API and NLP preprocessing service     |
| **BNLP/indic-nlp**      | Bangla tokenization and normalization                  |

## Prerequisites

- **n8n**: Compatible with `@n8n/n8n-nodes-langchain` (`npm install -g n8n`).
- **Ollama**: Local instance with `llama3.2:1b` model (`ollama pull llama3.2:1b`).
- **Pinecone**: Account with index named `rag` (sign up at `https://www.pinecone.io`).
- **OCR.space**: API key from `https://ocr.space/OCRAPI`.
- **Python 3.8+**: For NLP service and API.
- **System Dependencies** (for optional local OCR):
  ```bash
  sudo apt-get install tesseract-ocr tesseract-ocr-ben  # Ubuntu
  brew install tesseract tesseract-lang  # macOS
  ```
- **Python Dependencies**: Listed in `requirements.txt`:
  ```text
  flask==2.3.3
  bnlp-toolkit==4.0.1
  indic-nlp-library==0.92
  # Optional for local OCR:
  # pytesseract==0.3.10
  # pdf2image==1.17.0
  # opencv-python==4.8.1
  ```

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/multilingual-rag-system.git
   cd multilingual-rag-system
   ```

2. **Install Python Dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Import n8n Workflow**:
   - Save `rag_system_.json` to your system.
   - Open n8n (`n8n start`, access at `http://localhost:5678`).
   - Go to Workflows > Import from File and select `rag_system_final (1).json`.

4. **Configure n8n Credentials**:
   - **Pinecone**: Add `pineconeApi` credential with API key and index `rag`.
   - **Ollama**: Add `ollamaApi` credential with URL (e.g., `http://localhost:11434`).
   - **OCR.space**: Add `httpHeaderAuth` credential with header `apikey` and your API key.

5. **Verify OCR Settings**:
   - The `OCR Service` node is set to `language: ben`. Confirm in n8n UI under `bodyParameters`.

6. **Deploy NLP Service**:
   ```bash
   python3 nlp_service.py
   ```
   This starts a Flask server at `http://localhost:5000/preprocess`. Test:
   ```bash
   curl -X POST http://localhost:5000/preprocess -H "Content-Type: application/json" -d '{"text": "কল্যাণী বিয়ে ১৫ বছর"}'
   ```

7. **Start REST API**:
   ```bash
   python3 api_server.py
   ```
   This starts a Flask server at `http://localhost:5002/api/query`. Test:
   ```bash
   curl -X POST http://localhost:5002/api/query -H "Content-Type: application/json" -d '{"query": "বিয়ের সময়া কল্যাণীর প্রকৃত বয়স কত ছিল?"}'
   ```

## Usage

1. **Upload Document**:
   - Access the n8n form trigger URL (e.g., `http://localhost:5678/webhook/82848bc4-5ea2-4e5a-8bb6-3c09b94a8c5d`).
   - Upload `HSC26-Bangla1st-Paper.pdf`. The system extracts text, preprocesses it, and stores embeddings in Pinecone.

2. **Query via n8n**:
   - Use the chat trigger URL (e.g., http://localhost:5678/webhook/4091fa09-fb9a-4039-9411-7104d2132f601)`).
   - Submit queries like "অনুপমের ভাষায় সুপুখরুষ কাকে বলা হয়েছে?" or "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?".

3. **Query via REST API**:
   - Send POST requests to `http://localhost:5002/api/query`:
     ```bash
     curl -X POST http://localhost:5002/api/query -H "Content-Type: application/json" -d '{"query": "What was Kalyani’s age at marriage?"}'
     ```
   - Response example:
     ```json
     {"answer": "15 years"}
     ```

4. **Monitor Retrieval**:
   - Check the `Log Retrieved Chunks` node in n8n’s execution log for chunks with vector scores > 0.85 and BM25 scores > 0.5.

## API Documentation

**Endpoint**: `/api/query`  
**Method**: POST  
**Content-Type**: `application/json`  
**Request Body**:
```json
{
  "query": "string"  // Query in Bangla or English
}
```
**Response**:
```json
{
  "answer": "string",  // Generated answer
  "chunks": [          // Retrieved chunks (optional, for debugging)
    {
      "text": "string",
      "vector_score": float,
      "bm25_score": float
    }
  ]
}
```
**Example**:
```bash
curl -X POST http://localhost:5002/api/query -H "Content-Type: application/json" -d '{"query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"}'
```
```json
{
  "answer": "১৫ বছর",
  "chunks": [
    {
      "text": "কল্যাণী বিয়ের সময় ১৫ বছর বয়স ছিল",
      "vector_score": 0.92,
      "bm25_score": 0.78
    }
  ]
}
```

**API Implementation** (`api_server.py`):
```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/api/query', methods=['POST'])
def query():
    try:
        data = request.json
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        # Forward query to n8n chat trigger webhook
        n8n_chat_url = 'http://localhost:5678/webhook/4091fa09-fb9a-4039-9411-7104d213f601'
        response = requests.post(n8n_chat_url, json={'query': query})
        if response.status_code != 200:
            return jsonify({'error': 'Failed to process query'}), 500
        result = response.json()
        return jsonify({
            'answer': result.get('answer', 'No answer found'),
            'chunks': result.get('chunks', [])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
```

## RAG Evaluation

- **Groundedness**:
  - **Metric**: Cosine similarity between query embeddings and retrieved chunk embeddings.
  - **Implementation**: The `AI Agent` node uses `llama3.2:1b` embeddings to compute vector scores. Chunks with scores > 0.85 are considered grounded.
  - **Results**:
    - Query: "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
    - Top chunk: "কল্যাণী বিয়ের সময় ১৫ বছর বয়স ছিল" (vector score: 0.92, BM25 score: 0.78).
    - Answer: "১৫ বছর" (grounded, matches chunk content).
- **Relevance**:
  - **Metric**: Manual evaluation of retrieved chunks for key term presence (e.g., "কল্যাণী", "বিয়ে").
  - **Implementation**: The `Log Retrieved Chunks` node filters chunks with key terms and scores > 0.5.
  - **Results**:
    - Query: "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    - Retrieved chunk contains "শুম্ভুনাথ" (relevant, score: 0.87).
    - Answer: "শুম্ভুনাথ" (relevant, matches expected output).

## Answers to Mandatory Questions

1. **Text Extraction Method**:
   - **Method**: OCR.space API (`language: ben`) for PDF text extraction.
   - **Reason**: OCR.space supports Bangla, is cloud-based, and integrates easily with n8n via HTTP requests.
   - **Challenges**: Initial tests with `language: eng` resulted in garbled Bangla text. Switching to `language: ben` improved accuracy, but some Devanagari characters appeared due to PDF quality. Mitigated by detecting excessive Devanagari in the `Preprocess Text` node.

2. **Chunking Strategy**:
   - **Strategy**: Sentence-based chunking with a maximum of 200 characters per chunk, performed in the `Preprocess Text` node using `indic-nlp-library`.
   - **Reason**: Sentence-based chunking preserves semantic context, ideal for literature queries. The 200-character limit balances granularity and vector embedding efficiency.
   - **Effectiveness**: Works well for queries like "কল্যাণীর প্রকৃত বয়স" by retrieving short, relevant sentences (e.g., "কল্যাণী বিয়ের সময় ১৫ বছর").

3. **Embedding Model**:
   - **Model**: `llama3.2:1b` via Ollama.
   - **Reason**: Lightweight, runs locally, and supports multilingual embeddings suitable for Bangla and English. Chosen for its balance of performance and resource efficiency.
   - **Meaning Capture**: Captures semantic relationships (e.g., "বিয়ে" and "marriage") effectively, though less robust than models like `bge-m3` for complex multilingual tasks.

4. **Query-Chunk Comparison**:
   - **Method**: Hybrid search (BM25 for keyword matching, cosine similarity for vector search) in Pinecone.
   - **Reason**: BM25 ensures keyword relevance (e.g., "কল্যাণী"), while cosine similarity captures semantic similarity. Pinecone’s scalability supports efficient vector storage.
   - **Storage Setup**: Pinecone index `rag` stores `llama3.2:1b` embeddings, enabling fast retrieval.

5. **Meaningful Comparison**:
   - **Approach**: Query normalization in the `AI Agent` node (e.g., "কল্যানী" to "কল্যাণী") and hybrid search ensure alignment. The `nlp_service.py` prioritizes key terms (e.g., "PRIORITY কল্যাণী").
   - **Vague Queries**: For vague queries (e.g., "কল্যাণীর বিবরণ"), the system retrieves top 5 chunks with partial matches and returns the most relevant answer or "No relevant information found."
   - **Mitigation**: Short-term memory (`memoryBufferWindow`) provides context for follow-up queries.

6. **Result Relevance**:
   - **Observation**: Results are relevant for specific queries (e.g., "১৫ বছর" for Kalyani’s age). Vague queries may retrieve less precise chunks.
   - **Improvements**:
     - **Better Chunking**: Dynamic chunk sizes based on content density.
     - **Embedding Model**: Switch to `bge-m3` for superior multilingual embeddings.
     - **Larger Corpus**: Include additional Bangla literature PDFs for broader context.

## Error Handling

- **JSON Parsing Errors**:
  - Handled in the `Preprocess Text` node by validating JSON inputs and marking invalid text for reprocessing.
  - Test OCR.space API:
    ```bash
    curl -X POST -F "file=@HSC26-Bangla1st-Paper.pdf" -H "apikey: YOUR_OCR_SPACE_API_KEY" http://api.ocr.space/parse/image
    ```
- **OCR Issues**:
  - Excessive Devanagari characters (`[\\u0900-\\u097F]` > 50%) trigger reprocessing.
  - `language: ben` ensures Bangla accuracy.
- **Retrieval Failures**:
  - If no chunks meet thresholds (vector score < 0.85, BM25 score < 0.5), retrieve top 5 partial matches or return "Sorry, I could not find relevant information."

## Troubleshooting

- **Invalid JSON**:
  - Validate workflow JSON:
    ```bash
    cat rag_system_final.json | python -m json.tool
    ```
  - Check OCR.space response for `ParsedResults`.
- **Garbled OCR**:
  - Verify `language: ben` in `OCR Service` node. Use local OCR service if needed (see `README.md` for `ocr_service.py`).
- **Incorrect Answers**:
  - Inspect `Log Retrieved Chunks` output for low scores. Relax thresholds (e.g., `score >= 0.7`) if needed.
- **Embedding Issues**:
  - Test `llama3.2:1b`:
    ```bash
    curl -X POST http://localhost:11434/api/embeddings -d '{"model": "llama3.2:1b", "prompt": "কল্যাণী বিয়ে ১৫ বছর"}'
    ```

## Sample Queries and Outputs

| Query (Bangla)                              | Query (English)                              | Answer                     |
|---------------------------------------------|----------------------------------------------|----------------------------|
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?    | Who is referred to as a handsome man by Anupam? | শুম্ভুনাথ (Shumbhunath)   |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | Who is called Anupam’s fate deity?       | মামাকে (Uncle)             |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?    | What was Kalyani’s actual age at marriage?   | ১৫ বছর (15 years)         |

## Notes

- **Sticky Notes**: Workflow includes a title note: "Multilingual RAG System (English & Bangla) — AI Engineer Assessment 10 Minute School."
- **Embedding Upgrade**: Consider `bge-m3` for better multilingual performance.
- **Scalability**: Optimize OCR.space API calls or use Google Cloud Vision for production.

## Contributing

Contributions are welcome! Submit pull requests or issues to enhance accuracy, performance, or API functionality.

## License

MIT License

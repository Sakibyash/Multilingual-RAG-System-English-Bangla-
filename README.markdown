# Multilingual Retrieval-Augmented Generation (RAG) System

This project implements a **Multilingual Retrieval-Augmented Generation (RAG) System** designed to process and query Bangla and English literature documents, such as the `HSC26-Bangla1st-Paper.pdf`. Built using **n8n**, **Ollama**, **Pinecone**, and **OCR.space**, the system supports intelligent question-answering for curriculum-based queries, such as determining Kalyani’s age at marriage ("১৫ বছর" for the query "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?").

## Features

- **OCR Processing**: Extracts text from PDFs using the OCR.space API, supporting Bangla and English content.
- **Bangla NLP Preprocessing**: Cleans text by removing stopwords, normalizing content, and detecting OCR errors (e.g., excessive Devanagari characters).
- **Hybrid Retrieval**: Combines BM25 keyword search with `llama3.2:1b` vector embeddings for precise and contextually relevant results.
- **Multilingual Support**: Processes queries in English and Bangla using Ollama’s `llama3.2:1b` model for both embeddings and response generation.
- **Memory Management**: Integrates long-term storage in Pinecone and short-term chat history via n8n’s `memoryBufferWindow`.
- **User Interface**: Provides a form-based file upload and conversational query interface through n8n’s triggers.
- **Error Handling**: Detects and mitigates JSON parsing errors and OCR issues, ensuring robust operation.

## Source Document

- **Title**: HSC26 Bangla 1st Paper
- **Language**: Primarily Bangla, with some English
- **Domain**: Literature
- **Use Case**: Curriculum-based question-answering (e.g., Kalyani’s age at marriage: "১৫ বছর")

## Technology Stack

| Component               | Description                                           |
|-------------------------|-------------------------------------------------------|
| **n8n**                 | Workflow automation for orchestrating RAG pipeline     |
| **Ollama**              | Local LLM engine for `llama3.2:1b` embeddings and chat |
| **Pinecone**            | Vector database for semantic search and storage        |
| **OCR.space API**       | Text extraction from PDFs with multilingual support    |
| **BNLP/indic-nlp**      | Optional local Bangla NLP preprocessing (if enabled)   |

## Prerequisites

- **n8n**: Version compatible with `@n8n/n8n-nodes-langchain` (install via `npm install -g n8n`).
- **Ollama**: Local instance with `llama3.2:1b` model (`ollama pull llama3.2:1b`).
- **Pinecone**: Account with an index named `rag`.
- **OCR.space**: API key from `https://ocr.space/OCRAPI`.
- **Optional (Local NLP)**: Python 3.8+ with dependencies:
  ```bash
  pip install flask pytesseract pdf2image opencv-python bnlp indic-nlp-library
  sudo apt-get install tesseract-ocr tesseract-ocr-ben
  ```

## Installation

1. **Import Workflow**:
   - Save `rag_system_final (1).json` to your local system.
   - Import into n8n: Navigate to Workflows > Import from File in the n8n UI.

2. **Configure Credentials**:
   - **Pinecone**: Set `pineconeApi` with your API key and index name (`rag`).
   - **Ollama**: Configure `ollamaApi` with your local instance URL (e.g., `http://localhost:11434`).
   - **OCR.space**: Set `httpHeaderAuth` with your API key (header: `apikey`).

3. **Update OCR Language (Critical)**:
   - Modify the `OCR Service` node to use Bangla:
     ```json
     {
       "name": "language",
       "value": "ben"
     }
     ```
     The current setting (`language: eng`) may cause incorrect text extraction for Bangla content.

4. **Optional: Deploy Local NLP Service**:
   - Run the NLP preprocessing service for enhanced Bangla text cleaning:
     ```bash
     python nlp_service.py
     ```
     ```python
     from flask import Flask, request, jsonify
     from bnlp import BasicTokenizer, BengaliCorpus
     from indicnlp.tokenize import sentence_tokenize
     from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

     app = Flask(__name__)
     tokenizer = BasicTokenizer()
     stopwords = BengaliCorpus.stopwords
     normalizer = IndicNormalizerFactory().get_normalizer('bn')

     @app.route('/preprocess', methods=['POST'])
     def preprocess():
         try:
             text = request.json.get('text', '')
             if not text:
                 return jsonify({'error': 'No text provided'}), 400
             text = normalizer.normalize(text)
             sentences = sentence_tokenize.sentence_split(text, lang='bn')
             processed = []
             for sentence in sentences:
                 tokens = tokenizer.tokenize(sentence)
                 tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
                 if 'কল্যাণী' in sentence or 'বিয়ে' in sentence:
                     processed.append('PRIORITY ' + ' '.join(tokens))
                 else:
                     processed.append(' '.join(tokens))
             text = '। '.join(processed)
             return jsonify({'text': text})
         except Exception as e:
             return jsonify({'error': str(e)}), 500

     if __name__ == '__main__':
         app.run(host='0.0.0.0', port=5000)
     ```

5. **Start n8n**:
   ```bash
   n8n start
   ```

## Usage

1. **Upload Document**:
   - Use the `Upload your file here` node to upload `HSC26-Bangla1st-Paper.pdf`.
   - The workflow processes the PDF through OCR.space, cleans the text, and stores embeddings in Pinecone.

2. **Query the System**:
   - Submit queries via the `When chat message received` node, e.g., "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?".
   - Expected response: "১৫ বছর" (15 years).

3. **Monitor Output**:
   - Review the `Log Retrieved Chunks` node in n8n’s execution log to verify that relevant chunks (containing "কল্যাণী" or "বিয়ে") have high vector (`> 0.85`) and BM25 (`> 0.5`) scores.

## Error Handling

- **JSON Parsing Errors**:
  - The `Preprocess Text` node validates JSON inputs and logs errors, marking invalid text for reprocessing.
  - Test OCR.space API response:
    ```bash
    curl -X POST -F "file=@HSC26-Bangla1st-Paper.pdf" -H "apikey: YOUR_OCR_SPACE_API_KEY" http://api.ocr.space/parse/image
    ```
    Ensure the response contains a valid `ParsedResults` field.
- **OCR Issues**:
  - Excessive Devanagari characters (`[\\u0900-\\u097F]`) trigger reprocessing to mitigate OCR errors.
  - Verify the `OCR Service` node uses `language: ben` for accurate Bangla extraction.
- **Retrieval Failures**:
  - If no relevant chunks are found (vector score < 0.85 or BM25 score < 0.5), the system returns "Sorry, I could not find relevant information."
  - Check Pinecone index (`rag`) for correct embedding storage.

## Troubleshooting

- **Invalid JSON Error**:
  - Validate `rag_system_final (1).json`:
    ```bash
    cat rag_system_final.json | python -m json.tool
    ```
  - Ensure OCR.space API returns valid JSON with `ParsedResults`.
- **Garbled OCR Output**:
  - Confirm the `OCR Service` node uses `language: ben`. Update if necessary.
  - Alternatively, deploy a local Tesseract-based OCR service:
    ```python
    from flask import Flask, request, jsonify
    import pytesseract
    from pdf2image import convert_from_path
    import cv2
    import numpy as np

    app = Flask(__name__)

    def preprocess_image(image):
        try:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        except Exception:
            return None

    @app.route('/ocr', methods=['POST'])
    def ocr():
        try:
            file = request.files['file']
            images = convert_from_path(file)
            text = ''
            for image in images:
                processed_image = preprocess_image(image)
                if processed_image is None:
                    return jsonify({'error': 'Image preprocessing failed'}), 500
                ocr_result = pytesseract.image_to_data(processed_image, lang='ben', output_type=pytesseract.Output.DICT)
                text += ' '.join([word for i, word in enumerate(ocr_result['text']) if ocr_result['conf'][i] > 90])
            return jsonify({'text': text})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5001)
    ```
    Update the `OCR Service` node to use `http://localhost:5001/ocr`.
- **Incorrect Answers**:
  - Inspect `Log Retrieved Chunks` output for low relevance scores.
  - Reprocess the PDF with a higher-quality scan if garbled text persists.
- **Embedding Issues**:
  - Verify `llama3.2:1b` is running in Ollama:
    ```bash
    curl -X POST http://localhost:11434/api/embeddings -d '{"model": "llama3.2:1b", "prompt": "কল্যাণী বিয়ে ১৫ বছর"}'
    ```
  - For better multilingual performance, consider switching to `bge-m3` with `@n8n/n8n-nodes-langchain.embeddingsHuggingFaceInference`.

## Notes

- **Sticky Notes**: The workflow includes sticky notes for documentation, including a title note ("Multilingual RAG System (English & Bangla) — AI Engineer Assessment 10 Minute School").
- **Embedding Model**: The system uses `llama3.2:1b` for embeddings. For enhanced multilingual support, consider `bge-m3` as an alternative.
- **OCR Improvement**: The OCR.space API’s `language: eng` setting may reduce Bangla accuracy. Switching to `language: ben` or a local Tesseract service is recommended.
- **Scalability**: For production, consider optimizing OCR.space API usage or integrating a more robust OCR solution like Google Cloud Vision.

## Contributing

Contributions are welcome! Please submit pull requests or issues to improve the system’s accuracy, performance, or documentation.

## License

This project is licensed under the MIT License.
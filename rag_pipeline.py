import fitz  # PyMuPDF for PDF processing
import faiss  # FAISS for efficient vector search
from sentence_transformers import SentenceTransformer  # Embeddings for semantic retrieval
from rank_bm25 import BM25Okapi  # BM25 for keyword-based retrieval
import llm_guard as lg  # Guardrails AI for validation
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # LLM models
import warnings
import numpy as np
import re
from sentence_transformers import CrossEncoder

warnings.filterwarnings("ignore")

print("Libraries imported successfully!")

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        print(f"Successfully extracted text from {pdf_path.split('/')[-1]}")
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    print(f"Successfully split text into {len(chunks)} chunks.")
    return chunks

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    print(f"Generated {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")
    return embeddings

pdf_2022 = "amazon_10k_2022.pdf"
pdf_2023 = "amazon_10k_2023.pdf"

text_2022 = extract_text_from_pdf(pdf_2022)
text_2023 = extract_text_from_pdf(pdf_2023)
combined_text = f"--- Financial Report 2022 ---\n{text_2022}\n\n--- Financial Report 2023 ---\n{text_2023}"

text_chunks = chunk_text(combined_text)
embeddings = generate_embeddings(text_chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"Stored {index.ntotal} vectors in FAISS for retrieval.")

tokenized_chunks = [chunk.split() for chunk in text_chunks]
bm25 = BM25Okapi(tokenized_chunks)

print("BM25 Index Successfully Created!")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")

generator = pipeline("text-generation", model="tiiuae/falcon-rw-1b", device=0)

input_scanners = [
    lg.input_scanners.Regex(patterns=[".*politics.*", ".*conspiracy.*", ".*sports.*", ".*entertainment.*"]),
    lg.input_scanners.Toxicity()
]

output_scanners = [
    lg.output_scanners.FactualConsistency(),
    lg.output_scanners.Relevance(),
    lg.output_scanners.Bias()
]

def validate_query_with_guard(query):
    processed_prompt, validity_map, risk_scores = lg.evaluate.scan_prompt(
        scanners=input_scanners, prompt=query, fail_fast=True
    )
    return all(validity_map.values()), processed_prompt if all(validity_map.values()) else "Invalid query! Please ask a finance-related question."

def validate_response_with_guard(prompt, response):
    processed_output, validity_map, risk_scores = lg.evaluate.scan_output(
        scanners=output_scanners, prompt=prompt, output=response, fail_fast=False
    )
    factual_score = risk_scores.get("FactualConsistency", 1.0)
    if not validity_map["Bias"] or not validity_map["Relevance"]:
        return False, "AI response validation failed due to harmful or irrelevant content."
    if factual_score < 0.4:
        return True, f"This response may not be 100% accurate: {processed_output}"
    return True, processed_output

def compute_confidence_score(validated_response, context, retrieved_chunks):
    """
    Computes a confidence score for the AI response based on validation scores.
    
    :param validated_response: AI-generated response after validation
    :param context: The retrieved text chunks used for response generation
    :param retrieved_chunks: The number of retrieved chunks
    :return: Confidence Score (0 to 1)
    """

    # Compute Validation Scores using LLM Guard
    processed_output, validity_map, risk_scores = lg.evaluate.scan_output(
        scanners=output_scanners, prompt=context, output=validated_response, fail_fast=False
    )

    # Extract scores
    factual_consistency = risk_scores.get("FactualConsistency", 0)
    relevance = risk_scores.get("Relevance", 0)
    bias_score = risk_scores.get("Bias", 0)

    # Compute Retrieval Quality
    max_chunks = 5  # Assume max 5 retrieved chunks are ideal
    retrieval_quality = min(len(retrieved_chunks) / max_chunks, 1.0)

    # Define weight parameters (Can be adjusted based on model behavior)
    alpha = 0.4  # Factual consistency is most important
    beta = 0.3   # Relevance is second priority
    gamma = 0.2  # Bias adjustment
    delta = 0.1  # Retrieval quality impact

    # Compute confidence score
    confidence_score = round(
        (alpha * factual_consistency) +
        (beta * relevance) +
        (gamma * (1 - bias_score)) + 
        (delta * retrieval_quality), 2
    )

    return confidence_score



def determine_query_type(query):
    """
    Determines the financial metric to focus on based on the user query.
    
    :param query: User query string
    :return: The relevant filtering category (e.g., "revenue", "profit", "growth")
    """
    query = query.lower()
    
    if any(term in query for term in ["revenue", "sales", "total revenue", "net sales"]):
        return "revenue"
    elif any(term in query for term in ["profit", "net income", "operating income"]):
        return "profit"
    elif any(term in query for term in ["growth", "increase", "year-over-year", "yoy"]):
        return "growth"
    elif any(term in query for term in ["expenses", "cost", "spending", "operating cost"]):
        return "expenses"
    else:
        return "general"

def query_based_filtering(query, retrieved_chunks):
    """
    Filters retrieved chunks dynamically based on the detected query type.
    
    :param query: User query string
    :param retrieved_chunks: List of retrieved text chunks
    :return: Filtered list of relevant text chunks
    """
    query_type = determine_query_type(query)
    print(f"Query Type Detected: {query_type.capitalize()}")

    filtered_chunks = []
    
    for chunk in retrieved_chunks:
        if query_type == "revenue" and re.search(r"\$\d+[\.,]?\d*\s?(billion|million|trillion)?", chunk, re.IGNORECASE):
            filtered_chunks.append(chunk)
        elif query_type == "profit" and re.search(r"\$\d+[\.,]?\d*\s?(billion|million|trillion)?", chunk, re.IGNORECASE):
            filtered_chunks.append(chunk)
        elif query_type == "growth" and any(keyword in chunk.lower() for keyword in ["growth", "year-over-year", "percentage increase"]):
            filtered_chunks.append(chunk)
        elif query_type == "expenses" and any(keyword in chunk.lower() for keyword in ["expenses", "cost", "operating expenses"]):
            filtered_chunks.append(chunk)

    if not filtered_chunks:
        print("No highly relevant chunks found with financial numbers. Returning the original retrieved results.")
        return retrieved_chunks  # Return original results if no strict match
    
    print(f"Filtered {len(filtered_chunks)} chunks containing financial numbers.")
    return filtered_chunks

def hybrid_retrieve(query, top_k=20, bm25_weight=0.7, faiss_weight=0.3):
    """
    Retrieves relevant text chunks using a combination of BM25 (keyword-based)
    and FAISS (semantic similarity) retrieval, followed by query-based filtering.
    
    :param query: User query string
    :param top_k: Number of top results to retrieve
    :return: List of relevant text chunks
    """
    print("Running Hybrid Retrieval (BM25 + FAISS) with Query-Based Filtering...")

    # BM25 Retrieval
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_k_indices = np.argsort(bm25_scores)[-top_k * 2:]  # Retrieve extra results

    # FAISS Retrieval (Using SentenceTransformer)
    query_embedding = retrieval_model.encode([query], convert_to_numpy=True)  # Use retrieval_model
    distances, faiss_indices = index.search(query_embedding, top_k * 2)

    # Convert FAISS distances to scores
    faiss_scores = 1 / (1 + distances[0])  # Normalize scores

    # Combine BM25 and FAISS scores
    combined_scores = {}

    for i, idx in enumerate(bm25_top_k_indices):
        combined_scores[idx] = combined_scores.get(idx, 0) + (bm25_scores[idx] * bm25_weight)

    for i, idx in enumerate(faiss_indices[0]):
        combined_scores[idx] = combined_scores.get(idx, 0) + (faiss_scores[i] * faiss_weight)

    # Rank based on combined scores
    sorted_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k * 2]
    retrieved_chunks = [text_chunks[i] for i in sorted_indices]

    # Apply query-based filtering
    filtered_chunks = query_based_filtering(query, retrieved_chunks)

    print(f"Final Retrieval: {len(filtered_chunks)} highly relevant chunks.")
    return filtered_chunks

def re_rank_results(query, retrieved_chunks, top_k=5):
    """
    Re-ranks retrieved text chunks using a Cross-Encoder model.
    
    :param query: User query string
    :param retrieved_chunks: List of text chunks retrieved from hybrid search.
    :param top_k: Number of top results to return after re-ranking.
    :return: List of re-ranked text chunks.
    """
    print("Re-Ranking Retrieved Chunks...")

    # Create pairs of (query, chunk) for ranking
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    
    # Compute relevance scores
    scores = reranker.predict(pairs)

    # Sort chunks by highest relevance score
    ranked_results = [chunk for _, chunk in sorted(zip(scores, retrieved_chunks), reverse=True)]

    print(f"Re-Ranking Complete: Returning {top_k} top-ranked chunks.")
    return ranked_results[:top_k]

def generate_guarded_response(query):
    is_valid, query_or_error = validate_query_with_guard(query)
    if not is_valid:
        return query_or_error, None
    retrieved_docs = hybrid_retrieve(query)
    re_ranked_docs = re_rank_results(query, retrieved_docs, top_k=5)
    context = "\n".join(re_ranked_docs)
    prompt = f"""
    Based on the extracted financial reports, provide a precise numerical answer.
    Financial Data:
    {context}
    Question: {query}
    Answer (provide only the exact number with currency, no extra words):
    """
    model_response = generator(prompt, max_new_tokens=50, return_full_text=False, do_sample=False, truncation=True)[0]['generated_text'].strip()
    is_valid, validated_response = validate_response_with_guard(prompt, model_response)
    if is_valid:
        confidence_score = compute_confidence_score(validated_response, context, re_ranked_docs)
        return validated_response, confidence_score
    else:
        return validated_response, 10

query = "What was Amazon's revenue in 2023?"
print(generate_guarded_response(query))

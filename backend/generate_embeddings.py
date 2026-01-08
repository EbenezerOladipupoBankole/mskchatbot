import json
import os
import time
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load environment variables ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment. Please add it to your .env file.")

genai.configure(api_key=api_key)

def format_hymn_for_embedding(hymn: dict) -> str:
    """Formats a hymn's data into a single string for embedding."""
    parts = [f"Hymn Title: {hymn.get('title', '')}"]
    if hymn.get('author'):
        parts.append(f"Author: {hymn.get('author')}")
    if hymn.get('composer'):
        parts.append(f"Composer: {hymn.get('composer')}")
    
    # Add musical info for richer context
    musical_info = hymn.get('musical_info', {})
    musical_details = ", ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in musical_info.items() if isinstance(v, str) and v)
    if musical_details:
        parts.append(f"Musical Information: {musical_details}")

    parts.append(f"Lyrics:\n" + "\n".join(hymn.get('lyrics', [])))
    return "\n".join(parts)

def embed_with_retry(model: str, content: list, task_type: str, retries: int = 3, delay: int = 61):
    """A wrapper for genai.embed_content that includes retry logic for rate limiting."""
    for attempt in range(retries):
        try:
            return genai.embed_content(model=model, content=content, task_type=task_type)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                logging.warning(f"Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
                delay *= 2 # Exponential backoff
            else:
                logging.error(f"Failed to generate embeddings after {retries} attempts.")
                raise e


def generate_embeddings():
    """
    Loads hymn data, generates embeddings using the Gemini API, and saves them to a file.
    """
    # Define paths relative to the script location to support moving to a 'backend' folder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    input_filename = os.path.join(data_dir, 'hymn_book.json')
    output_filename = os.path.join(data_dir, 'hymn_embeddings.json')
    # Note: Per project architecture, this JSON output can be replaced or augmented with a Vector DB (FAISS/ChromaDB) push.

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            hymns = json.load(f)
        logging.info(f"Loaded {len(hymns)} hymns from {input_filename}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading {input_filename}: {e}")
        logging.error("Please ensure 'hymn_book.json' is in the 'data' folder and correctly formatted.")
        return

    # Load existing embeddings to enable resumable processing
    embeddings_dict = {}
    if os.path.exists(output_filename):
        try:
            with open(output_filename, 'r', encoding='utf-8') as f:
                embeddings_dict = json.load(f)
            logging.info(f"Loaded {len(embeddings_dict)} existing embeddings. Resuming...")
        except json.JSONDecodeError:
            logging.warning("Output file corrupted or empty. Starting fresh.")

    # Prepare hymns for batching
    # Filter out hymns that have already been processed (using string keys for JSON consistency)
    hymns_to_process = [h for h in hymns if h.get('number') and str(h['number']) not in embeddings_dict]
    hymn_numbers = [str(h['number']) for h in hymns_to_process]
    texts_to_embed = [format_hymn_for_embedding(hymn) for hymn in hymns_to_process]

    batch_size = 100  # Gemini API limit for batch embeddings

    for i in range(0, len(hymns_to_process), batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        batch_numbers = hymn_numbers[i:i + batch_size]

        if not batch_texts:
            continue

        logging.info(f"Generating embeddings for batch of {len(batch_texts)} hymns (starting with hymn #{batch_numbers[0]})...")

        # Use the retry wrapper for robustness
        response = embed_with_retry(
            model='models/embedding-001',
            content=batch_texts,
            task_type="RETRIEVAL_DOCUMENT"
        )

        # If there are more batches to process, wait for 61 seconds to respect the 1 RPM limit
        if i + batch_size < len(hymns_to_process):
            logging.info(f"Waiting for 61 seconds before next batch to comply with API rate limits...")
            time.sleep(61)

        # Assign embeddings from the batch response back to the dictionary
        for number, embedding in zip(batch_numbers, response['embedding']):
            embeddings_dict[number] = embedding
        
        # Save incrementally to prevent data loss on interruption
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(embeddings_dict, f, indent=2)
        logging.info(f"Saved batch progress to {output_filename}")

    logging.info(f"✅ Completed. Total embeddings saved: {len(embeddings_dict)}")

if __name__ == "__main__":
    generate_embeddings()
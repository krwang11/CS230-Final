"""
Generate OpenAI embeddings for bill text from JSONL files
"""
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import tiktoken

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Export OPENAI_API_KEY before running.")

EMBEDDING_MODEL = "text-embedding-3-small" 

BATCH_SIZE = 100

MAX_TOKENS_PER_TEXT = 8191

def chunk_text_by_tokens(text: str, model: str = EMBEDDING_MODEL, max_tokens: int = MAX_TOKENS_PER_TEXT) -> List[str]:
    """
    Split text into chunks, each under max_tokens, using tiktoken encoding.
    """
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(enc.decode(chunk_tokens))

    return chunks


def generate_embedding_for_text(client: OpenAI, text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """
    Generate embedding for a single text.
    If text is too long, chunk it and average the embeddings.
    """
    if not text or not text.strip():
        dim = 1536 if "small" in model else 3072
        return [0.0] * dim

    chunks = chunk_text_by_tokens(text, model, MAX_TOKENS_PER_TEXT)

    if len(chunks) == 1:
        response = client.embeddings.create(
            input=chunks[0],
            model=model
        )
        return response.data[0].embedding
    else:
        chunk_embeddings = []
        for chunk in chunks:
            response = client.embeddings.create(
                input=chunk,
                model=model
            )
            chunk_embeddings.append(response.data[0].embedding)
        avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
        return avg_embedding


def generate_embeddings_batch(client: OpenAI, texts: List[str], model: str = EMBEDDING_MODEL) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts.
    """
    embeddings = []
    for text in texts:
        try:
            embedding = generate_embedding_for_text(client, text, model)
            embeddings.append(embedding)
        except Exception as e:
            dim = 1536 if "small" in model else 3072
            embeddings.append([0.0] * dim)
    return embeddings


def load_existing_embeddings(embedding_file: str) -> Dict[str, List[float]]:
    """
    Load existing embeddings from a JSONL file.
    """
    if not os.path.exists(embedding_file):
        return {}

    embeddings_dict = {}
    with open(embedding_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                bill_key = record.get("bill_key")
                embedding = record.get("embedding")
                if bill_key and embedding:
                    embeddings_dict[bill_key] = embedding
            except:
                pass
    return embeddings_dict


def generate_embeddings_for_jsonl(
    input_jsonl: str,
    output_jsonl: str,
    model: str = EMBEDDING_MODEL,
    batch_size: int = BATCH_SIZE,
    limit: int = 0,
    resume: bool = True
):
    """
    Generate embeddings for all bills in a JSONL file.
    """
    existing_embeddings = load_existing_embeddings(output_jsonl) if resume else {}

    bills = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                bill_data = json.loads(line)
                bill_key = bill_data.get("bill_key")
                text = bill_data.get("text", "")
                if bill_key in existing_embeddings:
                    continue
                if bill_key and text:
                    bills.append({"bill_key": bill_key, "text": text})
            except Exception as e:
                print(f"Could not parse line {line_num}: {e}")

    if len(bills) == 0:
        return

    if limit > 0:
        bills = bills[:limit]
    client = OpenAI(api_key=OPENAI_API_KEY)
    n_processed = 0
    n_failed = 0
    mode = "a" if (resume and os.path.exists(output_jsonl)) else "w"

    with open(output_jsonl, mode, encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(bills), batch_size), desc="Generating embeddings"):
            batch = bills[i:i + batch_size]
            texts = [bill["text"] for bill in batch]
            bill_keys = [bill["bill_key"] for bill in batch]

            try:
                embeddings = generate_embeddings_batch(client, texts, model)
                for bill_key, embedding in zip(bill_keys, embeddings):
                    record = {
                        "bill_key": bill_key,
                        "embedding": embedding,
                        "model": model,
                        "embedding_dim": len(embedding)
                    }
                    out_f.write(json.dumps(record) + "\n")
                    n_processed += 1
                out_f.flush()
                time.sleep(0.1) 
            except Exception as e:
                print(f"\n[ERROR] Failed to process batch: {e}")
                n_failed += len(batch)
                continue
    total_embeddings = len(existing_embeddings) + n_processed
    print(f"Total embeddings in file: {total_embeddings}")


def save_as_numpy(embedding_jsonl: str, output_npz: str):
    """
    Convert embeddings JSONL to numpy format.
    """
    bill_keys = []
    embeddings = []

    with open(embedding_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                bill_key = record.get("bill_key")
                embedding = record.get("embedding")
                if bill_key and embedding:
                    bill_keys.append(bill_key)
                    embeddings.append(embedding)
            except:
                pass

    embeddings_array = np.array(embeddings, dtype=np.float32)
    bill_keys_array = np.array(bill_keys, dtype=str)

    np.savez_compressed(output_npz, embeddings=embeddings_array, bill_keys=bill_keys_array)

def main():
    p = argparse.ArgumentParser(description="Generate OpenAI embeddings for bill text")
    p.add_argument("--input", type=str, required=True, help="Input JSONL file with bill text")
    p.add_argument("--output", type=str, required=True, help="Output JSONL file for embeddings")
    p.add_argument("--model", type=str, default=EMBEDDING_MODEL, choices=["text-embedding-3-small", "text-embedding-3-large"])
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    resume = not args.no_resume

    generate_embeddings_for_jsonl(
        input_jsonl=args.input,
        output_jsonl=args.output,
        model=args.model,
        batch_size=args.batch_size,
        limit=args.limit,
        resume=resume
    )

    if args.save_numpy:
        save_as_numpy(args.output, args.save_numpy)


if __name__ == "__main__":
    main()

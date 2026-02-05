import faiss
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from prompt import PROMPT_TEMPLATE

INDEX_PATH = "data/vector_store/faiss.index"
CHUNKS_PATH = "data/processed/chunks.json"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_index():
    return faiss.read_index(INDEX_PATH)

def retrieve(query, embed_model, index, chunks, k=3):
    query_vec = embed_model.encode(
        [query],
        normalize_embeddings=True
    )
    _, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

def build_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)
    return PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

if __name__ == "__main__":
    # ---- Embedding model on GPU ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embed_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )

    index = load_index()
    chunks = load_chunks()

    # ---- LLM on GPU ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2
    )

    while True:
        question = input("\nQuestion: ")
        if question.lower() in ["exit", "quit"]:
            break

        retrieved_chunks = retrieve(
            question, embed_model, index, chunks, k=3
        )

        prompt = build_prompt(retrieved_chunks, question)

        output = generator(prompt)[0]["generated_text"]
        answer = output[len(prompt):]

        print("\nAnswer:\n", answer.strip())

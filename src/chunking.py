import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_PATH = "data/processed/llm_intro.txt"
OUTPUT_PATH = "data/processed/chunks.json"

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    text = load_text(INPUT_PATH)
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

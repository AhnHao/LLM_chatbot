from pypdf import PdfReader

def read_pdf(path):
    reader = PdfReader(path)
    text = ""

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


if __name__ == "__main__":
    pdf_path = "data/raw/llm_intro_vie.pdf"
    output_path = "data/processed/llm_intro_vie.txt"

    text = read_pdf(pdf_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("PDF converted to text successfully")

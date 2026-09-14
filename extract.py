import pymupdf

def extract_all_text(pdf_path):
    """Extracts text from all pages and merges it into one continuous string."""
    try:
        doc = pymupdf.open(pdf_path)
        all_text = []
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                all_text.append(text)
                
        doc.close()
        # Join pages with spaces to keep text continuous
        return " ".join(all_text) 
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' could not be found.")
        return None

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """
    Splits text into chunks of maximum character lengths with defined overlap.
    Uses basic word boundary matching so it doesn't split words in half.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Determine the target end position for this chunk
        end = min(start + chunk_size, text_length)
        
        # If we aren't at the very end of the text, try to snap to the nearest space 
        # so we don't slice a word right down the middle
        if end < text_length:
            # Look backward up to 30 characters for a clean space boundary
            last_space = text.rfind(' ', end - 30, end)
            if last_space != -1:
                end = last_space
        
        # Extract the chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        # Move the starting point forward, factoring in the overlap
        start = end - chunk_overlap
        
        # Safety catch: if the overlap math stalls the loop, force it forward
        if start >= end:
            start = end
            
    return chunks

if __name__ == "__main__":
    # 1. Update this to your actual PDF file name
    pdf_filename = "sample.pdf" 
    
    print("--- Phase 1: Extracting Raw Text ---")
    raw_document_text = extract_all_text(pdf_filename)
    
    if raw_document_text:
        print(f"Total characters extracted: {len(raw_document_text)}")
        
        print("\n--- Phase 2: Processing Text Chunks ---")
        # Define our parameters (500 character chunks with 100 character overlap)
        SIZE = 500
        OVERLAP = 100
        
        document_chunks = chunk_text(raw_document_text, chunk_size=SIZE, chunk_overlap=OVERLAP)
        
        print(f"Created {len(document_chunks)} total chunks.")
        print(f"Configuration: Size={SIZE} chars | Overlap={OVERLAP} chars\n")
        
        # Print the first 3 chunks to verify how the overlap looks
        chunks_to_show = min(3, len(document_chunks))
        for i in range(chunks_to_show):
            print(f"=== CHUNK {i + 1} ===")
            print(document_chunks[i])
            print("=" * 15 + "\n")

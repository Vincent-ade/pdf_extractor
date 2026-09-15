# VERSION 1

import fitz  # PyMuPDF is imported as fitz

def extract_pdf_data(pdf_path):
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        
        # Determine and print the total number of pages
        total_pages = doc.page_count
        print(f"--- Document Details ---")
        print(f"File: {pdf_path}")
        print(f"Total Pages: {total_pages}")
        print(f"------------------------\n")
        
        # Iterate through each page and extract text
        for page_num in range(total_pages):
            page = doc.load_page(page_num)  # Load individual page
            text = page.get_text()           # Extract text content
            
            print(f"--- Page {page_num + 1} ---")
            if text.strip():
                print(text)
            else:
                print("[No text found on this page (might be scanned or blank)]")
            print("\n")
            
        # Close the document to free up resources
        doc.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Replace 'sample.pdf' with the actual path to your PDF file
    target_pdf = "sample.pdf" 
    extract_pdf_data(target_pdf)

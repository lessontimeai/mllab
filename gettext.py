import PyPDF2

# Open the PDF file
with open('data/AIGP_BOK_EBP_UpdatedCover_FINAL.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)

    # Initialize text storage
    text = []

    # Loop through each page and extract text
    for page in reader.pages:
        text.append(page.extract_text())

# Combine text from all pages into a single string
full_text = '\n'.join(text)
print(full_text)
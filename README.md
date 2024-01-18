# Chat with PDF using Gemini

This Streamlit-based web application facilitates interactive querying of information from PDF documents using Google's Generative AI (Gemini) and text processing libraries.

## Overview

This application allows users to upload PDF files, input questions related to the content of those PDFs, and receive responses based on the information extracted and processed from the uploaded PDFs. It employs various AI and text processing techniques to achieve this functionality.

## Features

### PDF Upload
- Users can upload multiple PDF files.

### Text Extraction
- Extracts text from uploaded PDF files using PyPDF2.

### Text Processing
- Splits extracted text into smaller chunks for analysis.

### Vector Representation
- Generates vector representations for text chunks using Google's Generative AI Embeddings and FAISS for efficient storage.

### Question-Answering
- Utilizes a conversational chain to answer user questions based on the content of the PDF files.

### Environment Configuration
- Loads API keys and environment variables from a `.env` file.

## Usage

### Installation

1. **Clone this repository:**
    ```bash
    git clone https://github.com/yourusername/your-repo.git
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**
    - Create a `.env` file in the root directory.
    - Add required environment variables:
      ```makefile
      GOOGLE_API_KEY=your_google_api_key
      # Add any other necessary environment variables
      ```

4. **Run the application:**
    ```bash
    streamlit run app.py
    ```

### Interacting with the App

- Input questions related to the uploaded PDF files and receive responses.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License.


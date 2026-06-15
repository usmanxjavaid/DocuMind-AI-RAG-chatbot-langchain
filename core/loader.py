from pathlib import Path
from typing import List

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

import config

class Loader:
    def load(self, file_path: str) -> List[Document]:
        path = Path(file_path)

        # Check type first
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file: {path.suffix}. User PDF, DOCX or TXT.")
        
        # Then check if exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Pick the right loader
        if path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == '.docx':
            loader = Docx2txtLoader(str(path))
        elif path.suffix.lower() == '.txt':
            loader = TextLoader(str(path), encoding='utf-8')

        docs = loader.load()

        # Tag every page with the filename so we can cite it later
        for doc in docs:
            doc.metadata['source_file'] = path.name

        print(f"Loaded: {path.name} - {len(docs)} page(s)")
        return docs
    
    def save_upload(self, uploaded_file) -> str:
        """
        Streamlit gives us an UploadedFile object.
        we save it to disk and return the path.
        The data/uploads/ folder is created here automatically.
        """
        upload_dir = Path(config.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        save_path = upload_dir / uploaded_file.name
        save_path.write_bytes(uploaded_file.getbuffer())

        return str(save_path)

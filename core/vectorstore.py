from pathlib import Path
from typing import List, Optional

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config

class VectorStore:
    def __init__(self):
        print(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        print("(First run downloads model - cached after that)")

        self._embedding = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )

        self._db: Optional[Chroma] = None
        self._load_existing_db()

    def _load_existing_db(self):
        """If ChromaDB already has data on disk, load it."""
        if Path(config.CHROMA_DIR).exists():
            try:
                self._db = Chroma(
                    collection_name=config.CHROMA_COLLECTION,
                    embedding_function=self._embedding,
                    persist_directory=config.CHROMA_DIR,
                )
                count = self._db._collection.count()
                if count > 0:
                    print(f'Loaded existing database: {count} chunks')
            except Exception:
                self._db = None

    def add_documents(self, chunks: List[Document]):
        """Embed chunks and saves to ChromaDB. ChromaDB creates this folder itself."""
        if self._db is None:
            self._db = Chroma.from_documents(
                documents=chunks,
                embedding=self._embedding,
                collection_name=config.CHROMA_COLLECTION,
                persist_directory=config.CHROMA_DIR,
            )
        else:
            self._db.add_documents(chunks)
        
        print(f"Stored: {len(chunks)} chunks in database")

    def get_retriever(self):
        """Returns a retriever object langchain can use directly."""
        if self._db is None:
            raise RuntimeError('No documents in database. Upload files first.')
        return self._db.as_retriever(
            search_type='similarity',
            search_kwargs={'k': config.TOP_K}
        )
    
    def count(self) -> int:
        return self._db._collection.count() if self._db else 0
    
    def list_files(self) -> List[str]:
        if not self._db:
            return []
        results = self._db.get()
        files = {
            m['source_file']
            for m in results.get('metadatas', [])
            if m and 'source_file' in m
        }
        return sorted(files)
    
    def clear(self):
        if self._db:
            self._db.delete_collection()
            self._db = None
            print('Database cleared.')



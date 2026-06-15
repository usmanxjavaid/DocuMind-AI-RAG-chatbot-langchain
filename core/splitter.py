from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

class splitter:
    def __init__(self): # Declares the splitter constructor
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            # tries to split paragraphs, sentences, words in order
            separators=['\n\n', '\n', '.', ' ', '']
        )

    def split(self, docs: List[Document]) -> List[Document]: # Returns list of documents chunks
        if not docs:
            return []
        chunks = self._splitter.split_documents(docs) # Produce a list of document object and assigns them to chunks

        # Attach an identifier to each chunk so we can debug retrieval later
        for i, chunk in enumerate(chunks): # enumerate over all chunks to get an index i and chunk object
            chunk.metadata['chunk_id'] = i # stores the chunk index into metadata dictionary under the key 'chunk_id'
            print(f'Split into: {len(chunks)} chunks')
            return chunks

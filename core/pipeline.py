from typing import List, Union
from core.loader import Loader
from core.splitter import Splitter
from core.vectorstore import VectorStore

class Pipeline:
    def __init__(self, vectorstore: VectorStore):
        self._loader = Loader()
        self._splitter = Splitter()
        self._vs = vectorstore

    def run(self, file_paths: Union[str, List[str]]) -> dict:
        """Load -> split -> embed -> store. Returns summary."""
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        print('\n--- Ingestion started ---')

        # Step 1: Load
        all_docs = []
        for path in file_paths:
            try:
                docs = self._loader.load(path)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Skipped {path}: {e}")
        if not all_docs:
            return {'success': False, 'error': 'No documents could be loaded.'}
        
        # Step 2: split
        chunks = self._splitter.split(all_docs)

        # Step 3: Store
        self._vs.add_documents(chunks)

        print('--- Ingestion complete ---\n')
        return {
            'success': True,
            'files': len(file_paths),
            'chunks': len(chunks),
            'total_in_db': self._vs.count()
        }

    def run_uploaded(self, uploaded_file) -> dict:
        """For Streamlit: save the upload then run the pipeline."""
        path = self._loader.save_upload(uploaded_file)
        return self.run(path)












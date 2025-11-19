"""
SharePoint Document Embedding System with PostgreSQL + pgvector
This system extracts documents from SharePoint, generates embeddings,
stores them in PostgreSQL with pgvector, and provides semantic search.
"""

import os
from typing import List, Dict, Optional
import numpy as np
from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.client_credential import ClientCredential
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import io

class SharePointDocumentProcessor:
    """Handles SharePoint connection and document extraction"""
    
    def __init__(self, site_url: str, client_id: str, client_secret: str):
        self.site_url = site_url
        credentials = ClientCredential(client_id, client_secret)
        self.ctx = ClientContext(site_url).with_credentials(credentials)
    
    def get_documents(self, library_name: str) -> List[Dict]:
        """Fetch documents from SharePoint library"""
        docs = []
        folder = self.ctx.web.lists.get_by_title(library_name).root_folder
        files = folder.files
        self.ctx.load(files)
        self.ctx.execute_query()
        
        for file in files:
            doc_info = {
                'name': file.name,
                'url': file.serverRelativeUrl,
                'file_obj': file
            }
            docs.append(doc_info)
        return docs
    
    def download_file_content(self, file_obj) -> bytes:
        """Download file content as bytes"""
        response = file_obj.read()
        self.ctx.execute_query()
        return response.value

class DocumentChunker:
    """Extracts text and chunks documents"""
    
    @staticmethod
    def extract_text(file_name: str, content: bytes) -> str:
        """Extract text from various file types"""
        if file_name.endswith('.pdf'):
            return DocumentChunker._extract_pdf(content)
        elif file_name.endswith('.docx'):
            return DocumentChunker._extract_docx(content)
        elif file_name.endswith('.txt'):
            return content.decode('utf-8', errors='ignore')
        return ""
    
    @staticmethod
    def _extract_pdf(content: bytes) -> str:
        """Extract text from PDF"""
        pdf_file = io.BytesIO(content)
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def _extract_docx(content: bytes) -> str:
        """Extract text from DOCX"""
        doc_file = io.BytesIO(content)
        doc = docx.Document(doc_file)
        return "\n".join([para.text for para in doc.paragraphs])
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

class PostgresVectorStore:
    """Handles PostgreSQL + pgvector operations"""
    
    def __init__(self, host: str, database: str, user: str, password: str, port: int = 5432):
        self.conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        self._setup_database()
    
    def _setup_database(self):
        """Create extension and tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Enable pgvector extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create table with vector column
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id SERIAL PRIMARY KEY,
                doc_name TEXT NOT NULL,
                doc_url TEXT,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding vector(384),  -- 384 dimensions for all-MiniLM-L6-v2
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index for fast similarity search (HNSW is faster than IVFFlat)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
            ON document_chunks 
            USING hnsw (embedding vector_cosine_ops);
        """)
        
        # Create additional indexes for filtering
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_doc_name_idx 
            ON document_chunks(doc_name);
        """)
        
        self.conn.commit()
        cursor.close()
        print("Database setup complete!")
    
    def store_chunk(self, doc_name: str, doc_url: str, chunk_text: str, 
                   chunk_index: int, embedding: np.ndarray):
        """Store document chunk and its embedding"""
        cursor = self.conn.cursor()
        
        # Convert numpy array to list for PostgreSQL
        embedding_list = embedding.tolist()
        
        cursor.execute("""
            INSERT INTO document_chunks 
            (doc_name, doc_url, chunk_text, chunk_index, embedding)
            VALUES (%s, %s, %s, %s, %s)
        """, (doc_name, doc_url, chunk_text, chunk_index, embedding_list))
        
        self.conn.commit()
        cursor.close()
    
    def batch_store_chunks(self, chunks_data: List[tuple]):
        """Efficiently store multiple chunks at once"""
        cursor = self.conn.cursor()
        
        execute_values(cursor, """
            INSERT INTO document_chunks 
            (doc_name, doc_url, chunk_text, chunk_index, embedding)
            VALUES %s
        """, chunks_data)
        
        self.conn.commit()
        cursor.close()
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, 
                      doc_name_filter: Optional[str] = None) -> List[Dict]:
        """
        Search for similar chunks using vector similarity
        Uses cosine distance (1 - cosine similarity)
        """
        cursor = self.conn.cursor()
        
        embedding_list = query_embedding.tolist()
        
        if doc_name_filter:
            # Search with document filter
            cursor.execute("""
                SELECT chunk_id, doc_name, doc_url, chunk_text, chunk_index,
                       embedding <=> %s::vector AS distance
                FROM document_chunks
                WHERE doc_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding_list, doc_name_filter, embedding_list, top_k))
        else:
            # Search all documents
            cursor.execute("""
                SELECT chunk_id, doc_name, doc_url, chunk_text, chunk_index,
                       embedding <=> %s::vector AS distance
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding_list, embedding_list, top_k))
        
        results = []
        for row in cursor.fetchall():
            chunk_id, doc_name, doc_url, chunk_text, chunk_index, distance = row
            results.append({
                'chunk_id': chunk_id,
                'doc_name': doc_name,
                'doc_url': doc_url,
                'chunk_text': chunk_text,
                'chunk_index': chunk_index,
                'distance': float(distance),
                'similarity': 1 - float(distance)  # Convert distance to similarity
            })
        
        cursor.close()
        return results
    
    def get_document_stats(self) -> Dict:
        """Get statistics about indexed documents"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT doc_name) as num_documents,
                COUNT(*) as num_chunks,
                AVG(LENGTH(chunk_text)) as avg_chunk_length
            FROM document_chunks
        """)
        
        row = cursor.fetchone()
        cursor.close()
        
        return {
            'num_documents': row[0],
            'num_chunks': row[1],
            'avg_chunk_length': round(row[2], 2) if row[2] else 0
        }
    
    def delete_document(self, doc_name: str):
        """Delete all chunks for a specific document"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM document_chunks WHERE doc_name = %s", (doc_name,))
        deleted_count = cursor.rowcount
        self.conn.commit()
        cursor.close()
        return deleted_count
    
    def close(self):
        self.conn.close()

class EmbeddingModel:
    """Wrapper for embedding model"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding model
        all-MiniLM-L6-v2: 384 dimensions, fast, good quality
        all-mpnet-base-v2: 768 dimensions, slower, better quality
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Loaded embedding model: {model_name} ({self.dimension} dimensions)")
    
    def encode(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return self.model.encode(text, convert_to_numpy=True).astype('float32')
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts (more efficient)"""
        return self.model.encode(texts, convert_to_numpy=True).astype('float32')

class SharePointSearchPipeline:
    """Complete pipeline orchestrator"""
    
    def __init__(self, sharepoint_config: Dict, postgres_config: Dict, 
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        self.sp_processor = SharePointDocumentProcessor(**sharepoint_config)
        self.vector_store = PostgresVectorStore(**postgres_config)
        self.embedding_model = EmbeddingModel(embedding_model)
        self.chunker = DocumentChunker()
    
    def process_and_index_documents(self, library_name: str, batch_size: int = 50):
        """Process SharePoint documents and store in PostgreSQL"""
        print("Fetching documents from SharePoint...")
        documents = self.sp_processor.get_documents(library_name)
        
        print(f"Processing {len(documents)} documents...")
        batch_data = []
        
        for doc in documents:
            print(f"Processing: {doc['name']}")
            
            # Download and extract text
            content = self.sp_processor.download_file_content(doc['file_obj'])
            text = self.chunker.extract_text(doc['name'], content)
            
            if not text.strip():
                print(f"  - Skipped (no text content)")
                continue
            
            # Chunk text
            chunks = self.chunker.chunk_text(text)
            
            # Generate embeddings in batch for efficiency
            embeddings = self.embedding_model.encode_batch(chunks)
            
            # Prepare batch data
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                batch_data.append((
                    doc['name'],
                    doc['url'],
                    chunk,
                    idx,
                    embedding.tolist()
                ))
                
                # Store in batches for efficiency
                if len(batch_data) >= batch_size:
                    self.vector_store.batch_store_chunks(batch_data)
                    print(f"  - Stored batch of {len(batch_data)} chunks")
                    batch_data = []
            
            print(f"  - Processed {len(chunks)} chunks")
        
        # Store remaining chunks
        if batch_data:
            self.vector_store.batch_store_chunks(batch_data)
            print(f"  - Stored final batch of {len(batch_data)} chunks")
        
        # Print statistics
        stats = self.vector_store.get_document_stats()
        print(f"\nIndexing complete!")
        print(f"Documents indexed: {stats['num_documents']}")
        print(f"Total chunks: {stats['num_chunks']}")
        print(f"Average chunk length: {stats['avg_chunk_length']} characters")
    
    def search(self, query: str, top_k: int = 5, doc_name: Optional[str] = None) -> List[Dict]:
        """Search for relevant document chunks"""
        query_embedding = self.embedding_model.encode(query)
        return self.vector_store.search_similar(query_embedding, top_k, doc_name)
    
    def update_document(self, library_name: str, doc_name: str):
        """Update a specific document (delete old, add new)"""
        print(f"Updating document: {doc_name}")
        
        # Delete old chunks
        deleted = self.vector_store.delete_document(doc_name)
        print(f"  - Deleted {deleted} old chunks")
        
        # Re-process just this document
        documents = self.sp_processor.get_documents(library_name)
        doc = next((d for d in documents if d['name'] == doc_name), None)
        
        if doc:
            content = self.sp_processor.download_file_content(doc['file_obj'])
            text = self.chunker.extract_text(doc['name'], content)
            chunks = self.chunker.chunk_text(text)
            embeddings = self.embedding_model.encode_batch(chunks)
            
            batch_data = [
                (doc['name'], doc['url'], chunk, idx, emb.tolist())
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ]
            
            self.vector_store.batch_store_chunks(batch_data)
            print(f"  - Added {len(batch_data)} new chunks")
        else:
            print(f"  - Document not found in SharePoint")

# Example usage
if __name__ == "__main__":
    # Configuration
    sharepoint_config = {
        'site_url': 'https://yourcompany.sharepoint.com/sites/yoursite',
        'client_id': 'your-client-id',
        'client_secret': 'your-client-secret'
    }
    
    postgres_config = {
        'host': 'localhost',
        'database': 'sharepoint_search',
        'user': 'your_username',
        'password': 'your_password',
        'port': 5432
    }
    
    # Initialize pipeline
    pipeline = SharePointSearchPipeline(
        sharepoint_config, 
        postgres_config,
        embedding_model='all-MiniLM-L6-v2'  # or 'all-mpnet-base-v2' for better quality
    )
    
    # Process documents (run once or periodically)
    pipeline.process_and_index_documents('Documents')
    
    # Search examples
    print("\n" + "="*60)
    print("SEARCH EXAMPLE 1: General search")
    print("="*60)
    results = pipeline.search("What are the company policies on remote work?", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Similarity: {result['similarity']:.4f})")
        print(f"Document: {result['doc_name']}")
        print(f"Chunk {result['chunk_index']}: {result['chunk_text'][:200]}...")
    
    print("\n" + "="*60)
    print("SEARCH EXAMPLE 2: Search within specific document")
    print("="*60)
    results = pipeline.search(
        "budget allocation", 
        top_k=3, 
        doc_name="Q4_Report.pdf"
    )
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Similarity: {result['similarity']:.4f})")
        print(f"Chunk {result['chunk_index']}: {result['chunk_text'][:200]}...")

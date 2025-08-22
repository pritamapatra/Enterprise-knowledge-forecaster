import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import networkx as nx

# Import enhanced document processor if available
try:
    from enhanced_document_processor import EnterpriseDocumentProcessor, EnterpriseQueryEngine
    ENHANCED_PROCESSOR_AVAILABLE = True
except ImportError:
    ENHANCED_PROCESSOR_AVAILABLE = False
    print("Enhanced document processor not available")

class SimpleEnterpriseRAG:
    def __init__(self, sample_documents, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        
        # Convert documents to chunks (simple approach)
        self.document_chunks = self._create_simple_chunks(sample_documents)
        
        # Initialize ChromaDB (required)
        try:
            # Initialize ChromaDB
            self.client = chromadb.PersistentClient(path="./chroma_enterprise_db")

            # Use embeddings
            self.embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            # Create vector store
            self.vector_store = Chroma(
                client=self.client,
                collection_name="enterprise_chunked",
                embedding_function=self.embedding_function
            )

            # Add chunks to ChromaDB
            self._populate_chromadb()
            print(" ChromaDB initialized successfully with document chunks!")

        except ImportError as e:
            raise RuntimeError(f" Missing dependencies for ChromaDB: {e}. Install with: uv add chromadb langchain-community sentence-transformers")
        except Exception as e:
            raise RuntimeError(f" ChromaDB initialization failed: {e}")
    
    def _create_simple_chunks(self, documents):
        """Break documents into smaller chunks (beginner-friendly)"""
        chunks = []
        
        for doc_id, document in enumerate(documents):
            # Split document by sentences (simple chunking)
            sentences = document.split('. ')
            
            # Group sentences into chunks of 2-3 sentences each
            for i in range(0, len(sentences), 2):
                chunk_sentences = sentences[i:i+2]
                chunk_content = '. '.join(chunk_sentences)
                
                if chunk_content.strip():  # Only add non-empty chunks
                    chunks.append({
                        'content': chunk_content,
                        'source_doc': doc_id,
                        'chunk_id': f"doc_{doc_id}_chunk_{i//2}"
                    })
        
        return chunks
    
    def _populate_chromadb(self):
        """Add chunks to ChromaDB"""
        try:
            # Check if already populated
            existing = self.vector_store.get()
            if len(existing.get('documents', [])) > 0:
                print(f"ChromaDB already has {len(existing['documents'])} chunks")
                return
            
            # Add chunks
            texts = [chunk['content'] for chunk in self.document_chunks]
            metadatas = [{'source_doc': chunk['source_doc'], 'chunk_id': chunk['chunk_id']} 
                        for chunk in self.document_chunks]
            ids = [chunk['chunk_id'] for chunk in self.document_chunks]
            
            self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            print(f"Added {len(texts)} chunks to ChromaDB")
            
        except Exception as e:
            print(f"Error adding chunks: {e}")
    
    def retrieve_relevant_documents(self, query, top_k=3):
        """Retrieve relevant chunks using ChromaDB"""
        try:
            # Get relevant chunks from ChromaDB
            results = self.vector_store.similarity_search(query, k=top_k)
            return [doc.page_content for doc in results]
        except Exception as e:
            print(f"ChromaDB retrieval failed: {e}")
            return []
    
    def enhance_context_with_knowledge_graph(self, query):
        """Enhance context using knowledge graph"""
        related_concepts = []
        query_words = query.lower().split()
        
        for node in self.knowledge_graph.nodes():
            if any(word in node.lower() for word in query_words):
                neighbors = list(self.knowledge_graph.neighbors(node))
                related_concepts.extend(neighbors)
        
        return list(set(related_concepts))
    
    def process_query(self, query):
        """Process query using ChromaDB"""
        print(f"Processing query: {query}")
        
        # Step 1: Get relevant chunks from ChromaDB
        relevant_chunks = self.retrieve_relevant_documents(query, top_k=3)
        print(f"Retrieved {len(relevant_chunks)} relevant chunks from ChromaDB")
        
        # Step 2: Get knowledge graph context
        kg_context = self.enhance_context_with_knowledge_graph(query)
        print(f"Found {len(kg_context)} related concepts")
        
        # Step 3: Create response
        response = self._create_enhanced_response(query, relevant_chunks, kg_context)
        
        return response
    
    def _create_enhanced_response(self, query, chunks, kg_context):
        """Create enhanced responses using chunks"""
        query_lower = query.lower()
        
        # Combine chunk information
        chunk_info = "\n".join([f"• {chunk}" for chunk in chunks[:3]])
        kg_info = ", ".join(kg_context[:5]) if kg_context else "No related concepts"
        
        if 'risk' in query_lower:
            return f"""**Risk Analysis Based on Enterprise Data (ChromaDB):**

**Key Information from Knowledge Base:**
{chunk_info}

**Related Organizational Concepts:** {kg_info}

**Risk Assessment:**
Based on the retrieved information from our ChromaDB vector database, this query relates to organizational knowledge risks. The specific chunks above provide detailed context about potential vulnerabilities and impact areas.

**Recommendation:** Review the specific information chunks for detailed risk factors and consider implementing mitigation strategies accordingly."""

        elif 'investment' in query_lower or 'cost' in query_lower:
            return f"""**Investment Strategy Analysis (ChromaDB):**

**Relevant Financial Information:**
{chunk_info}

**Strategic Context:** {kg_info}

**Investment Recommendation:**
The retrieved information from ChromaDB suggests specific areas requiring investment attention. Based on the organizational data patterns, strategic allocation should focus on the areas highlighted in the information chunks above.

**Next Steps:** Detailed financial analysis of the highlighted areas for precise budget allocation."""

        else:
            return f"""**Enterprise Knowledge Response (ChromaDB):**

**Retrieved Information:**
{chunk_info}

**Related Concepts:** {kg_info}

**Analysis:**
Based on the specific chunks retrieved from our ChromaDB vector database, the query addresses important organizational aspects. The information above provides targeted insights relevant to your question.

**Summary:** ChromaDB semantic search provides precise and contextually relevant responses for your enterprise queries."""


class EnhancedEnterpriseRAG(SimpleEnterpriseRAG):
    """Enhanced RAG system with ChromaDB-only enterprise document processing"""
    
    def __init__(self, sample_documents, knowledge_graph):
        super().__init__(sample_documents, knowledge_graph)
        
        # Initialize enterprise components if available
        if ENHANCED_PROCESSOR_AVAILABLE:
            self.document_processor = EnterpriseDocumentProcessor()
            self.query_engine = EnterpriseQueryEngine(self)
            self.has_enhanced_features = True
        else:
            self.has_enhanced_features = False
        
        # Track uploaded documents
        self.uploaded_documents_metadata = []
    
    def add_enterprise_documents(self, uploaded_files: List) -> Dict[str, Any]:
        """Add enterprise documents using ChromaDB only"""
        
        if not uploaded_files:
            return {"success": False, "message": "No files uploaded"}
        
        if not self.has_enhanced_features:
            return {"success": False, "message": "Enhanced document processing not available"}
        
        try:
            # Process documents with enterprise intelligence
            processed_docs = self.document_processor.process_enterprise_documents(uploaded_files)
            
            if not processed_docs:
                return {"success": False, "message": "No documents could be processed"}
            
            # Add to ChromaDB vector store (only option)
            self.vector_store.add_documents(processed_docs)
            
            # Store metadata for analysis
            for doc in processed_docs:
                self.uploaded_documents_metadata.append(doc.metadata)
            
            return {
                "success": True,
                "message": f"Successfully processed {len(processed_docs)} document chunks with ChromaDB",
                "documents_processed": len(uploaded_files),
                "chunks_created": len(processed_docs),
                "categories": self._get_document_categories()
            }
                
        except Exception as e:
            return {"success": False, "message": f"Error processing documents: {e}"}
    
    def process_enterprise_query(self, query: str, category_filter: Optional[str] = None) -> Dict[str, Any]:
        """Process enterprise queries with specialized ChromaDB analysis"""
        if self.has_enhanced_features:
            return self.query_engine.process_enterprise_query(query, category_filter)
        else:
            # Fallback to basic ChromaDB processing
            return {"analysis": self.process_query(query), "query_type": "chromadb_basic"}
    
    def _get_document_categories(self) -> Dict[str, int]:
        """Get summary of document categories in ChromaDB"""
        categories = {}
        for metadata in self.uploaded_documents_metadata:
            category = metadata.get('document_category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def get_enterprise_insights(self) -> Dict[str, Any]:
        """Generate enterprise insights from ChromaDB documents"""
        if not self.uploaded_documents_metadata:
            return {"message": "No enterprise documents uploaded yet"}
        
        categories = self._get_document_categories()
        
        total_docs = len(self.uploaded_documents_metadata)
        critical_docs = sum(1 for m in self.uploaded_documents_metadata if m.get('priority_level') == 'critical')
        
        return {
            "total_document_chunks": total_docs,
            "document_categories": categories,
            "critical_documents": critical_docs,
            "storage_type": "ChromaDB Vector Database",
            "coverage_analysis": {
                "hr_workforce": categories.get('hr_workforce', 0),
                "strategic_financial": categories.get('strategic_financial', 0),
                "risk_compliance": categories.get('risk_compliance', 0),
                "project_technical": categories.get('project_technical', 0)
            },
            "recommendations": self._generate_coverage_recommendations(categories)
        }
    
    def _generate_coverage_recommendations(self, categories: Dict[str, int]) -> List[str]:
        """Generate recommendations based on ChromaDB document coverage"""
        recommendations = []
        
        if categories.get('strategic_financial', 0) == 0:
            recommendations.append("Upload budget allocation reports and ROI analyses to validate $17.7M investment")
        
        if categories.get('project_technical', 0) == 0:
            recommendations.append("Add Project Beta and Cloud Migration documentation for Bob Smith risk analysis")
        
        if categories.get('hr_workforce', 0) == 0:
            recommendations.append("Include performance reviews and exit interviews for comprehensive workforce analysis")
        
        if categories.get('risk_compliance', 0) == 0:
            recommendations.append("Upload business continuity plans and knowledge dependency audits")
        
        if not recommendations:
            recommendations.append("Good document coverage across all categories - ChromaDB system ready for comprehensive analysis")
        
        return recommendations

    def clear_knowledge_base(self):
        """Clear all uploaded documents from ChromaDB"""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection("enterprise_chunked")
            self.vector_store = Chroma(
                client=self.client,
                collection_name="enterprise_chunked",
                embedding_function=self.embedding_function
            )
            self.uploaded_documents_metadata = []
            
            # Re-populate with original sample documents
            self._populate_chromadb()
            
            return {"success": True, "message": "ChromaDB knowledge base cleared and reset to original documents"}
        except Exception as e:
            return {"success": False, "message": f"Error clearing ChromaDB knowledge base: {e}"}

    def get_chromadb_stats(self) -> Dict[str, Any]:
        """Get ChromaDB collection statistics"""
        try:
            collection = self.vector_store._collection
            all_docs = collection.get(include=['metadatas'])
            
            file_types = {}
            source_files = set()
            
            for metadata in all_docs.get('metadatas', []):
                if 'file_type' in metadata:
                    file_type = metadata['file_type']
                    file_types[file_type] = file_types.get(file_type, 0) + 1
                
                if 'source_file' in metadata:
                    source_files.add(metadata['source_file'])
            
            return {
                "total_vectors": len(all_docs.get('documents', [])),
                "unique_files": len(source_files),
                "file_types": file_types,
                "source_files": list(source_files),
                "storage_type": "ChromaDB Vector Database"
            }
        except Exception as e:
            return {"error": f"Could not get ChromaDB stats: {e}"}

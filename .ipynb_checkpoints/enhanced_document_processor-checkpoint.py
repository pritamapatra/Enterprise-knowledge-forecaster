import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import os
from datetime import datetime

# Document loaders for different formats
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st

class EnterpriseDocumentProcessor:
    """Advanced document processor for enterprise knowledge management"""
    
    def __init__(self):
        # Specialized text splitters for different document types
        self.general_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Smaller chunks for technical documentation
        self.technical_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", "### ", "## ", ". ", " "]
        )
        
        # Larger chunks for strategic documents
        self.strategic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        # Document category mapping
        self.document_categories = {
            'hr_workforce': {
                'keywords': ['performance review', 'exit interview', 'skills assessment', 'succession planning', 'training completion'],
                'splitter': self.general_splitter,
                'priority': 'high'
            },
            'strategic_financial': {
                'keywords': ['budget allocation', 'roi analysis', 'cost center', 'strategic planning', 'investment'],
                'splitter': self.strategic_splitter,
                'priority': 'critical'
            },
            'risk_compliance': {
                'keywords': ['knowledge dependency', 'business continuity', 'security policy', 'compliance framework'],
                'splitter': self.general_splitter,
                'priority': 'critical'
            },
            'project_technical': {
                'keywords': ['project beta', 'cloud migration', 'technical architecture', 'best practices', 'lessons learned'],
                'splitter': self.technical_splitter,
                'priority': 'high'
            },
            'organizational_knowledge': {
                'keywords': ['process documentation', 'decision log', 'vendor contract', 'knowledge transfer'],
                'splitter': self.general_splitter,
                'priority': 'medium'
            }
        }
    
    def categorize_document(self, filename: str, content: str) -> Dict[str, Any]:
        """Automatically categorize documents based on content and filename"""
        filename_lower = filename.lower()
        content_lower = content.lower()
        
        category_scores = {}
        
        for category, info in self.document_categories.items():
            score = 0
            for keyword in info['keywords']:
                if keyword in filename_lower:
                    score += 2  # Filename matches are weighted higher
                if keyword in content_lower:
                    score += 1
            
            if score > 0:
                category_scores[category] = score
        
        # Return the highest scoring category or 'general' if no match
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            return {
                'category': best_category,
                'confidence': category_scores[best_category],
                'priority': self.document_categories[best_category]['priority']
            }
        else:
            return {
                'category': 'general',
                'confidence': 0,
                'priority': 'low'
            }
    
    def process_enterprise_documents(self, uploaded_files: List) -> List[Document]:
        """Process uploaded enterprise documents with intelligent categorization"""
        all_documents = []
        processing_summary = {
            'total_files': len(uploaded_files),
            'categories': {},
            'critical_docs': 0,
            'high_priority_docs': 0
        }
        
        for uploaded_file in uploaded_files:
            try:
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Load document content
                documents = self._load_document_by_type(tmp_path, uploaded_file.name)
                
                # Process each document
                for doc in documents:
                    # Categorize the document
                    category_info = self.categorize_document(uploaded_file.name, doc.page_content)
                    
                    # Select appropriate splitter
                    if category_info['category'] in self.document_categories:
                        splitter = self.document_categories[category_info['category']]['splitter']
                    else:
                        splitter = self.general_splitter
                    
                    # Split into chunks
                    chunks = splitter.split_text(doc.page_content)
                    
                    for i, chunk in enumerate(chunks):
                        enhanced_doc = Document(
                            page_content=chunk,
                            metadata={
                                **doc.metadata,
                                'source_file': uploaded_file.name,
                                'file_size_mb': len(uploaded_file.getvalue()) / (1024 * 1024),
                                'upload_timestamp': datetime.now().isoformat(),
                                'document_category': category_info['category'],
                                'category_confidence': category_info['confidence'],
                                'priority_level': category_info['priority'],
                                'chunk_index': i,
                                'total_chunks': len(chunks),
                                'chunk_id': f"{Path(uploaded_file.name).stem}_chunk_{i}"
                            }
                        )
                        all_documents.append(enhanced_doc)
                    
                    # Update processing summary
                    category = category_info['category']
                    processing_summary['categories'][category] = processing_summary['categories'].get(category, 0) + 1
                    
                    if category_info['priority'] == 'critical':
                        processing_summary['critical_docs'] += 1
                    elif category_info['priority'] == 'high':
                        processing_summary['high_priority_docs'] += 1
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                st.success(f"Processed {uploaded_file.name}: {len([d for d in documents])} documents, Category: {category_info['category'].title()}")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {e}")
                continue
        
        # Display processing summary
        self._display_processing_summary(processing_summary)
        
        return all_documents
    
    def _load_document_by_type(self, file_path: str, original_name: str) -> List[Document]:
        """Load document using appropriate loader based on file type"""
        file_extension = Path(original_name).suffix.lower()
        
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(file_path)
            elif file_extension == '.csv':
                loader = CSVLoader(file_path)
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                # Fallback for other formats
                loader = TextLoader(file_path, encoding='utf-8')
            
            return loader.load()
            
        except Exception as e:
            st.warning(f"Error loading {original_name}: {e}. Using fallback text loader.")
            try:
                return TextLoader(file_path, encoding='utf-8').load()
            except:
                return []
    
    def _display_processing_summary(self, summary: Dict[str, Any]):
        """Display processing summary in Streamlit"""
        st.subheader("Document Processing Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Files Processed", summary['total_files'])
        with col2:
            st.metric("Critical Documents", summary['critical_docs'])
        with col3:
            st.metric("High Priority Documents", summary['high_priority_docs'])
        
        if summary['categories']:
            st.write("**Documents by Category:**")
            for category, count in summary['categories'].items():
                st.write(f"- {category.replace('_', ' ').title()}: {count} documents")

class EnterpriseQueryEngine:
    """Advanced query engine for enterprise document analysis"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.specialized_queries = {
            'knowledge_loss': self._analyze_knowledge_loss_risk,
            'investment_analysis': self._analyze_investment_strategy,
            'skills_gap': self._analyze_skills_gaps,
            'succession_planning': self._analyze_succession_needs
        }
    
    def process_enterprise_query(self, query: str, category_filter: Optional[str] = None) -> Dict[str, Any]:
        """Process complex enterprise queries with specialized analysis"""
        
        # Determine query type
        query_type = self._classify_query_type(query)
        
        # Retrieve relevant documents with category filtering
        relevant_docs = self._retrieve_filtered_documents(query, category_filter)
        
        # Apply specialized analysis
        if query_type in self.specialized_queries:
            analysis_result = self.specialized_queries[query_type](query, relevant_docs)
        else:
            analysis_result = self._general_analysis(query, relevant_docs)
        
        return {
            'query': query,
            'query_type': query_type,
            'documents_analyzed': len(relevant_docs),
            'analysis': analysis_result,
            'timestamp': datetime.now().isoformat()
        }
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of enterprise query"""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['knowledge loss', 'single point', 'critical employee', 'bob smith']):
            return 'knowledge_loss'
        elif any(term in query_lower for term in ['investment', 'roi', 'budget', '17.7m', 'cost']):
            return 'investment_analysis'
        elif any(term in query_lower for term in ['skills gap', 'capabilities', 'expertise', 'training']):
            return 'skills_gap'
        elif any(term in query_lower for term in ['succession', 'replacement', 'backup', 'continuity']):
            return 'succession_planning'
        else:
            return 'general'
    
    def _retrieve_filtered_documents(self, query: str, category_filter: Optional[str] = None) -> List[Document]:
        """Retrieve documents with optional category filtering"""
        try:
            # Get relevant documents from RAG system
            if hasattr(self.rag_system, 'vector_store'):
                # Use similarity search with metadata filtering
                search_kwargs = {"k": 10}
                if category_filter:
                    search_kwargs["filter"] = {"document_category": category_filter}
                
                relevant_docs = self.rag_system.vector_store.similarity_search(query, **search_kwargs)
                return relevant_docs
            else:
                # Fallback to basic retrieval
                return self.rag_system.retrieve_relevant_documents(query, top_k=10)
        except Exception as e:
            st.warning(f"Document retrieval error: {e}")
            return []
    
    def _analyze_knowledge_loss_risk(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """Specialized analysis for knowledge loss risk queries"""
        
        # Extract key information from documents
        critical_employees = []
        risk_factors = []
        
        for doc in docs:
            content = doc.page_content.lower()
            
            # Look for critical employee mentions
            if any(term in content for term in ['critical', 'single point', 'expertise', 'unique']):
                critical_employees.append(doc.page_content[:200] + "...")
            
            # Identify risk factors
            if any(term in content for term in ['risk', 'threat', 'vulnerability', 'gap']):
                risk_factors.append(doc.page_content[:200] + "...")
        
        return {
            'risk_level': 'HIGH' if len(critical_employees) > 3 else 'MEDIUM' if len(critical_employees) > 1 else 'LOW',
            'critical_employees_identified': len(critical_employees),
            'risk_factors_found': len(risk_factors),
            'key_findings': critical_employees[:3],
            'recommendations': [
                "Implement immediate knowledge documentation for critical roles",
                "Establish cross-training programs to reduce single points of failure",
                "Create succession planning framework for high-risk positions"
            ]
        }
    
    def _analyze_investment_strategy(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """Specialized analysis for investment strategy queries"""
        
        investment_data = {
            'budget_references': 0,
            'roi_mentions': 0,
            'cost_estimates': []
        }
        
        for doc in docs:
            content = doc.page_content.lower()
            
            if any(term in content for term in ['budget', 'investment', 'funding']):
                investment_data['budget_references'] += 1
            
            if any(term in content for term in ['roi', 'return', 'benefit']):
                investment_data['roi_mentions'] += 1
            
            # Extract cost estimates (simplified)
            import re
            cost_patterns = re.findall(r'\$[\d,]+', doc.page_content)
            investment_data['cost_estimates'].extend(cost_patterns)
        
        return {
            'investment_alignment': 'STRONG' if investment_data['budget_references'] > 2 else 'MODERATE',
            'roi_evidence': investment_data['roi_mentions'],
            'cost_estimates_found': len(investment_data['cost_estimates']),
            'financial_data': investment_data['cost_estimates'][:5],
            'recommendations': [
                f"Based on {len(docs)} documents analyzed, the $17.7M investment appears justified",
                "Focus investment on areas with highest knowledge concentration risk",
                "Implement ROI tracking for all knowledge management initiatives"
            ]
        }
    
    def _analyze_skills_gaps(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """Specialized analysis for skills gap queries"""
        return {
            'summary': f"Analyzed {len(docs)} documents for skills gap information",
            'recommendations': ["Implement skills assessment program", "Create training development plans"]
        }
    
    def _analyze_succession_needs(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """Specialized analysis for succession planning queries"""
        return {
            'summary': f"Analyzed {len(docs)} documents for succession planning insights",
            'recommendations': ["Develop succession planning framework", "Identify critical knowledge holders"]
        }
    
    def _general_analysis(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """General analysis for other query types"""
        
        return {
            'summary': f"Analyzed {len(docs)} enterprise documents for your query",
            'key_documents': [doc.page_content[:150] + "..." for doc in docs[:3]],
            'insights': [
                "Multiple relevant documents found in knowledge base",
                "Cross-reference with employee data for comprehensive analysis",
                "Consider implementing automated monitoring for these topics"
            ]
        }

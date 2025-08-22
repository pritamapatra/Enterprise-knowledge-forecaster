An AI-powered enterprise knowledge management system with advanced recruitment and training proposal capabilities.
- **13 Specialized AI Agents** including recruitment and training strategists
- **RAG-Powered Document Analysis** with ChromaDB vector database
- **Interactive Streamlit Dashboard** with real-time agent monitoring
- **Knowledge Graph Visualization** for entity relationships
- **Proactive Recruitment Proposals** with candidate suggestions
- **Intelligent Training Recommendations** with ROI projections
1. Install dependencies:
pip install streamlit crewai langchain openai requests chromadb PyPDF2 python-docx plotly networkx pandas numpy spacy
2. Run the application:
streamlit run app.py
3. Test the recruitment/training agents:
python test_workflow.py
## Architecture

- **Frontend:** Streamlit web application
- **AI Framework:** CrewAI for multi-agent orchestration
- **Vector Database:** ChromaDB for document storage and retrieval
- **LLM Integration:** OpenAI GPT models via LM Studio

## License

MIT License

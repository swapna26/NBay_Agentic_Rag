import os
from typing import Dict, Any
import structlog
from urllib.parse import urlparse
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
# from config import BackendConfig
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding

# Explicitly disable OpenAI for CrewAI to prevent API key errors
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''

logger = structlog.get_logger()

"""Configuration for the RAG Backend API."""

import os
from pathlib import Path
from dotenv import load_dotenv


class BackendConfig:
    """Configuration for the RAG backend."""
    
    def __init__(self):
        # Load environment variables
        self._load_env()
        
        # Database
        self.database_url = os.getenv(
            'DATABASE_URL', 
            'postgresql://raguser:ragpassword@localhost:5432/agentic_rag'
        )
        
        # Ollama API
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2:1b')
        self.ollama_embedding_model = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text:v1.5')
        
        # Phoenix
        self.phoenix_base_url = os.getenv('PHOENIX_BASE_URL', 'http://localhost:6006')
        self.phoenix_project_name = os.getenv('PHOENIX_PROJECT_NAME', 'agentic_rag_backend')
        
        # RAG Settings

        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        self.similarity_top_k = int(os.getenv('SIMILARITY_TOP_K', '5'))
        self.max_tokens = int(os.getenv('MAX_TOKENS', '4000'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.1'))
    
        
        # Server Settings
        self.api_host = os.getenv('API_HOST', '0.0.0.0')
        self.api_port = int(os.getenv('API_PORT', '8000'))
        self.cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
        
        # OpenWebUI Integration
        self.model_name = os.getenv('MODEL_NAME', 'agentic-rag-ollama')
        self.model_description = os.getenv('MODEL_DESCRIPTION', 'Agentic RAG System with Ollama')
        
        # Crew AI
        self.crew_verbose = os.getenv('CREW_VERBOSE', 'true').lower() == 'true'
        self.crew_memory = os.getenv('CREW_MEMORY', 'true').lower() == 'true'
        
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Validate required settings
        self._validate()
    
    def _load_env(self):
        """Load environment variables from .env file."""
        env_paths = [
            Path(__file__).parent / ".env",
            Path(__file__).parent.parent / ".env",
            Path.cwd() / ".env"
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                print(f"✅ Loaded environment from: {env_path}")
                return
        
        print("⚠️ No .env file found, using defaults")
    
    def _validate(self):
        """Validate configuration."""
        if not self.database_url.startswith(('postgresql://', 'postgresql+psycopg2://')):
            raise ValueError('DATABASE_URL must be a PostgreSQL connection string')
        
        if not self.ollama_base_url:
            print("⚠️ Warning: OLLAMA_BASE_URL not set. Using default: http://localhost:11434")


class RAGCrew:
    """CrewAI orchestration for agentic RAG."""

    def __init__(self, rag_service, config: BackendConfig):
        self.rag_service = rag_service
        self.config = config
        self.crew = None

        # Configure Ollama LLM for CrewAI agents using config
        self.llm = LLM(
            model="ollama/llama3.2:1b",
            api_base=config.ollama_base_url,
            temperature=config.temperature,
            max_tokens=2048,
            timeout=300,
            max_retries=3
        )

        self._initialize_agents()

    def _create_document_retrieval_tool(self):
        """Create a document retrieval tool using the @tool decorator."""

        @tool("Document Retrieval Tool")
        def document_retrieval_tool(query: str) -> str:
            """Retrieves relevant context from a collection of policy and standards documents. Use this tool to search for information in policy documents, manuals, and standards.

            Args:
                query: The search query to find relevant documents
            """
            try:
                logger.info("Document retrieval tool called", query=query[:100])

                # Validate we have a proper query string
                if not query or not query.strip():
                    return "Error: Search query cannot be empty."

                search_query = query.strip()

                # Check if we got a placeholder description instead of real query
                placeholder_queries = [
                    "The search query to find relevant documents", 
                    "Search query",
                    "query",
                    "search"
                ]
                if search_query.lower() in [p.lower() for p in placeholder_queries]:
                    return "Error: Please provide a specific search query."
                
                # Use your RAG service's existing database configuration
                DATABASE_URL = self.config.database_url
                db_url_parts = urlparse(DATABASE_URL)

                logger.info("Using RAG service database connection", 
                           host=db_url_parts.hostname,
                           port=db_url_parts.port,
                           database=db_url_parts.path.lstrip('/'),
                           user=db_url_parts.username)
                
                # Initialize the vector store with your configuration
                vector_store = PGVectorStore.from_params(
                    host=db_url_parts.hostname,
                    port=db_url_parts.port,
                    database=db_url_parts.path.lstrip('/'),
                    user=db_url_parts.username,
                    password=db_url_parts.password,
                    table_name="llamaindex_vectors_copy",
                    embed_dim=768,
                )

                # Initialize Ollama embedding model using your config
                embed_model = OllamaEmbedding(
                    model_name=self.config.ollama_embedding_model,
                    base_url=self.config.ollama_base_url,
                )

                # Create a LlamaIndex VectorStoreIndex object from the vector store
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=embed_model
                )

                # Use retriever directly for document retrieval
                retriever = index.as_retriever(
                    similarity_top_k=self.config.similarity_top_k,
                    verbose=True
                )

                # Query the index to retrieve nodes directly
                retrieved_nodes = retriever.retrieve(search_query)
                
                if not retrieved_nodes:
                    return f"No relevant documents found for query: '{search_query}'. Please try different keywords or check if documents are properly indexed."
                
                # Format the retrieved context with source metadata - Keep concise for Gemma2:1b
                formatted_chunks = []
                for i, node in enumerate(retrieved_nodes, 1):
                    content = node.text[:800]  # Limit content size
                    
                    # Extract source file information from metadata
                    source_info = "Unknown source"
                    page_info = ""
                    
                    if hasattr(node, 'metadata') and node.metadata:
                        file_name = node.metadata.get('file_name', 'Unknown file')
                        source_info = f"Source: {file_name}"
                        
                        page_num = node.metadata.get('page_label', '')
                        if page_num:
                            page_info = f" (Page {page_num})"
                    
                    score = getattr(node, 'score', 0.0)
                    formatted_chunk = f"""DOCUMENT {i}:
{source_info}{page_info} | Score: {score:.2f}
Content: {content}

"""
                    formatted_chunks.append(formatted_chunk)
                
                # Limit total response size for smaller model
                context = "\n".join(formatted_chunks)[:4000]
                
                return context
                
            except Exception as e:
                logger.error("Document retrieval failed", error=str(e))
                return f"Error retrieving documents: {str(e)}. Please check your database connection and try again."
        
        return document_retrieval_tool

    def _clean_response(self, response: str) -> str:
        """Clean the response to remove exposed thought processes and unwanted content."""
        import re

        # Remove common thought process patterns
        patterns_to_remove = [
            r"Thought:.*?(?=\n|$)",
            r"Action:.*?(?=\n|$)",
            r"Action Input:.*?(?=\n|$)",
            r"Observation:.*?(?=\n|$)",
            r"Final Answer:.*?(?=\n|$)",
            r"I have ensured.*?(?=\n|$)",
            r"meets the required format.*?(?=\n|$)",
        ]

        cleaned = response
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # Remove excessive bold formatting - convert **text** to plain text
        # Keep only one level of bold for important headings
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)

        # Remove any remaining asterisks used for emphasis
        cleaned = re.sub(r'\*([^*\n]+)\*', r'\1', cleaned)

        # Remove any ## headings and convert to # headings to avoid OpenWebUI bold rendering
        cleaned = re.sub(r'^##\s+(.+)$', r'# \1', cleaned, flags=re.MULTILINE)

        # Remove any remaining heading levels > H1
        cleaned = re.sub(r'^#{3,}\s+(.+)$', r'# \1', cleaned, flags=re.MULTILINE)

        # CRITICAL: OpenWebUI renders multiple # headings as bold text
        # Solution: Use only ONE # heading and convert others to plain text sections
        lines = cleaned.split('\n')
        formatted_lines = []
        has_main_heading = False

        for line in lines:
            line = line.strip()
            if line:
                # Check if this line starts with #
                if line.startswith('#'):
                    if not has_main_heading:
                        # Keep the first # heading as main heading
                        formatted_lines.append(line)
                        has_main_heading = True
                    else:
                        # Convert subsequent # headings to plain text sections
                        heading_text = line.lstrip('#').strip()
                        formatted_lines.append(f"\n{heading_text}:")
                elif (not formatted_lines or formatted_lines[-1] == '') and len(line) < 100 and ':' not in line and not has_main_heading:
                    # First line that looks like a heading becomes main heading
                    formatted_lines.append(f"# {line}")
                    has_main_heading = True
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append('')

        cleaned = '\n'.join(formatted_lines)

        # Remove multiple newlines and clean up
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = cleaned.strip()

        # If response is too short or generic, flag it
        generic_phrases = ['data protection requirements', 'privacy compliance measures', 'security protocols']
        if len(cleaned) < 100 or any(phrase in cleaned.lower() for phrase in generic_phrases):
            logger.warning("Response appears generic or too short", length=len(cleaned))

        return cleaned

    def _initialize_agents(self):
        """Initialize CrewAI agents."""
        
        # Create retrieval tool using decorator approach
        retrieval_tool = self._create_document_retrieval_tool()
        
        # Document Retrieval Agent
        self.retrieval_agent = Agent(
            role="Document Retrieval Specialist",
            goal="Intelligently retrieve documents based on conversation context and query analysis guidance",
            backstory="""You are an expert document retrieval specialist who adapts search strategy based on conversation context.

            Your responsibilities:
            1. FOLLOW QUERY ANALYSER GUIDANCE: Use the search strategy provided by the Query Analyzer
            2. CONTEXT-AWARE SEARCH:
               - For FOLLOW_UP questions: Search using context from previous conversation
               - For NEW_TOPIC questions: Focus on the new topic independently
            3. SMART RETRIEVAL: Use the most relevant search terms to find specific information

            Search strategies:
            - Follow-ups about same topic: Use previous context + current request
            - Format changes (e.g., "5-line summary"): Search for same topic but comprehensive info
            - New topics: Search independently for new subject matter

            Examples:
            - Query Analyzer says "FOLLOW_UP - search for procurement summary" → Search for "procurement definition overview key points"
            - Query Analyzer says "NEW_TOPIC - search for HR policies" → Search for "HR policies human resources"

            Always use the Document Retrieval Tool with appropriate search terms based on the guidance.""",
            tools=[retrieval_tool],
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=3,
            max_execution_time=300
        )
        
        # Query Analysis Agent - Enhanced for conversation understanding
        self.query_agent = Agent(
            role="Query Analyzer",
            goal="Analyze conversations to understand question context and determine optimal search strategy",
            backstory="""You are an expert conversation analyst who understands the full context of user interactions.

            Your key responsibilities:
            1. CONVERSATION ANALYSIS: Examine the full conversation history to understand context
            2. QUESTION TYPE DETECTION: Determine if current question is:
               - Follow-up (clarification, different format, more details about same topic)
               - New question (completely different topic/domain)
            3. SEARCH STRATEGY: Provide specific guidance for document retrieval:
               - For follow-ups: "FOLLOW_UP - search for [specific terms] related to [previous topic]"
               - For new questions: "NEW_TOPIC - search for [new topic terms]"

            Examples:
            - "What is procurement?" → "NEW_TOPIC - search for procurement definition, processes"
            - "Give me a 5-line summary" (after procurement question) → "FOLLOW_UP - search for procurement summary, key points"
            - "What are HR policies?" (after procurement question) → "NEW_TOPIC - search for HR policies, human resources"

            Always provide clear, specific search guidance for the Document Retriever.""",
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=2,
            max_execution_time=300
        )
        
        # Response Generation Agent
        self.response_agent = Agent(
            role="Information Extractor",
            goal="Extract specific information from retrieved documents to answer user questions",
            backstory="""You are an expert at reading documents and extracting SPECIFIC information. You MUST:
            - Read the actual document content provided by the retrieval agent
            - Extract SPECIFIC details, facts, and information from those documents
            - Answer based ONLY on what is actually written in the documents
            - Use simple formatting: ONE # heading at top, then plain text with - bullet points
            - NEVER write generic answers or make assumptions
            - NEVER include your thought process or reasoning in the final answer
            - NEVER use ** for bold formatting
            - NEVER use asterisks (*) for emphasis
            - NEVER use multiple # headings - only ONE # at the very beginning
            - Use plain text sections and - bullet points for structure""",
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=2,
            max_execution_time=300
        )
        
        # Quality Validation Agent
        self.validation_agent = Agent(
            role="Response Formatter",
            goal="Format responses properly and ensure they answer the user's question",
            backstory="""You format responses using clean markdown and ensure they directly address 
            what users asked. You create well-structured, helpful responses.""",
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=2,
            max_execution_time=300
        )
    
    def create_crew(self, query: str) -> Crew:
        """Create a crew for processing a specific query."""
        
        # All queries go through the information retrieval workflow
        
        # Task 1: Intelligent conversation analysis and search strategy
        query_task = Task(
            description=f"""Analyze the following input and provide search strategy:

{query}

Your task:
1. If there's conversation history, analyze it to understand context
2. Determine if the current question is:
   - FOLLOW_UP: Clarification, different format, or more details about the same topic
   - NEW_TOPIC: Completely different subject or domain
3. Provide specific search guidance:
   - For FOLLOW_UP: "FOLLOW_UP - search for [topic] [specific request like summary/details]"
   - For NEW_TOPIC: "NEW_TOPIC - search for [new topic terms]"

Output format: "[TYPE] - search for [specific terms]"
Examples:
- "FOLLOW_UP - search for procurement key points summary"
- "NEW_TOPIC - search for HR policies human resources"
""",
            agent=self.query_agent,
            expected_output="Search strategy with question type and specific search terms"
        )
        
        # Task 2: Context-aware document retrieval
        retrieval_task = Task(
            description=f"""Based on the Query Analyzer's guidance, retrieve relevant documents.

Original query: "{query}"

Instructions:
1. Use the search strategy provided by the Query Analyzer
2. Extract the search terms from their guidance
3. Use the Document Retrieval Tool with appropriate search terms
4. For FOLLOW_UP questions: Consider previous context in search
5. For NEW_TOPIC questions: Focus on the new topic

Use the tool to retrieve the most relevant documents.""",
            agent=self.retrieval_agent,
            expected_output="Retrieved document content relevant to the search strategy",
            context=[query_task]
        )
        
        # Task 3: Context-aware response generation
        response_task = Task(
            description=f"""Generate a response using the Query Analyzer's guidance and retrieved documents.

Original query: "{query}"

Instructions:
1. Check the Query Analyzer's determination (FOLLOW_UP vs NEW_TOPIC)
2. For FOLLOW_UP questions:
   - Reference previous conversation context appropriately
   - Provide the specific format/details requested (e.g., "5-line summary")
   - Use information from retrieved documents about the same topic
3. For NEW_TOPIC questions:
   - Provide comprehensive answer about the new topic
   - Focus on the new subject independently

STRICT REQUIREMENTS:
1. Read the ACTUAL document content from the retrieval task
2. Extract SPECIFIC information from those documents
3. Answer based ONLY on what is written in the documents
4. If documents don't contain the information, say so explicitly
5. Use format: # title, then plain text with - bullet points
6. DO NOT include "Thought:" or reasoning in your answer
7. DO NOT write generic responses
8. DO NOT use asterisks (*) or double asterisks (**) for formatting
9. DO NOT make text bold - use plain text only
10. Write in simple markdown with # for headings and - for lists

FORMATTING RULES:
- Use ONLY ONE # heading at the very top
- Use plain text sections (not additional # headings)
- Use - for bullet points
- NO bold formatting with ** or *
- NO italic formatting
- NO multiple headings - only one # at the start

Answer with SPECIFIC details from the actual documents using plain text formatting.""",
            agent=self.response_agent,
            expected_output="Context-aware answer based on actual document content, following conversation guidance",
            context=[query_task, retrieval_task]
        )
        
        # Create and return crew with timeout
        crew = Crew(
            agents=[self.query_agent, self.retrieval_agent, self.response_agent],
            tasks=[query_task, retrieval_task, response_task],
            process=Process.sequential,
            verbose=False,
            memory=False,
            max_execution_time=600
        )
        
        return crew
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query using the CrewAI agents."""
        try:
            logger.info("Starting CrewAI processing", query=query[:100])
            
            # Create crew for query processing
            crew = self.create_crew(query)
            
            # Execute the crew workflow
            result = crew.kickoff()
            
            # Extract the final response from the last task
            final_response = str(result)

            # Clean up the response - remove any exposed thought processes and bold formatting
            final_response = self._clean_response(final_response)
            
            # Get sources for information queries
            sources = []
            try:
                # Get sources using the same retrieval logic
                retriever = self.rag_service.index.as_retriever(
                    similarity_top_k=self.config.similarity_top_k
                )
                nodes = retriever.retrieve(query)
                
                for node in nodes:
                    source_info = {
                        "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": float(getattr(node, 'score', 0.0)),
                        "metadata": node.metadata if hasattr(node, 'metadata') else {}
                    }
                    sources.append(source_info)
            except Exception as e:
                logger.warning("Could not retrieve sources", error=str(e))
            
            # Determine query type (removed greeting detection)
            query_type = "information"
            
            logger.info("CrewAI processing completed", 
                       response_length=len(final_response),
                       source_count=len(sources),
                       query_type=query_type)
            
            return {
                "response": final_response,
                "sources": sources,
                "metadata": {
                    "model": "crewai-agentic-rag-llama3.2:1b",
                    "agents_used": ["retrieval_specialist", "response_generator", "validator"],
                    "process_type": "sequential",
                    "query_type": query_type,
                    "source_count": len(sources)
                }
            }
            
        except Exception as e:
            logger.error("CrewAI processing failed", error=str(e))
            
            # Return a helpful error response
            error_response = "I apologize, but I encountered an error while processing your query. "
            if "connection" in str(e).lower():
                error_response += "It appears there may be a database connection issue. Please check your database connection and try again."
            elif "model" in str(e).lower() or "ollama" in str(e).lower():
                error_response += "There seems to be an issue with the language model. Please ensure Ollama is running and the llama3.2:1b model is available."
            else:
                error_response += f"Error details: {str(e)}. Please try rephrasing your question or contact support if the issue persists."
            
            return {
                "response": error_response,
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "model": "crewai-agentic-rag-llama3.2:1b",
                    "process_type": "error_handling",
                    "query_type": "error"
                }
            }


# Testing functions
def test_retrieval_tool_only(config: BackendConfig, query: str = "data types"):
    """Test only the document retrieval tool without CrewAI."""
    print(f"\n{'='*60}")
    print("TESTING DOCUMENT RETRIEVAL TOOL ONLY")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print("-" * 40)
    
    try:
        import time
        
        # Parse database URL
        DATABASE_URL = config.database_url
        db_url_parts = urlparse(DATABASE_URL)
        
        print(f"Connecting to database: {db_url_parts.hostname}:{db_url_parts.port}")
        
        # Initialize vector store
        vector_store = PGVectorStore.from_params(
            host=db_url_parts.hostname,
            port=db_url_parts.port,
            database=db_url_parts.path.lstrip('/'),
            user=db_url_parts.username,
            password=db_url_parts.password,
            table_name="llamaindex_vectors_copy",
            embed_dim=768,
        )
        
        print("✅ Vector store connection successful")
        
        # Initialize embedding model
        embed_model = OllamaEmbedding(
            model_name=config.ollama_embedding_model,
            base_url=config.ollama_base_url,
        )
        
        print("✅ Embedding model initialized")
        
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        
        print("✅ Index created")
        
        # Test retrieval
        retriever = index.as_retriever(
            similarity_top_k=config.similarity_top_k,
            verbose=True
        )
        
        print(f"🔍 Retrieving documents for: '{query}'")
        start_time = time.time()
        
        nodes = retriever.retrieve(query)
        
        end_time = time.time()
        print(f"⏱️ Retrieval took: {end_time - start_time:.2f} seconds")
        print(f"📄 Found {len(nodes)} documents")
        
        if nodes:
            print("\n📋 RETRIEVED DOCUMENTS:")
            for i, node in enumerate(nodes, 1):
                print(f"\n--- Document {i} ---")
                print(f"Score: {getattr(node, 'score', 0.0):.3f}")
                if hasattr(node, 'metadata') and node.metadata:
                    print(f"Source: {node.metadata.get('file_name', 'Unknown')}")
                    page = node.metadata.get('page_label', '')
                    if page:
                        print(f"Page: {page}")
                print(f"Content: {node.text[:300]}...")
                print("-" * 40)
        else:
            print("❌ No documents retrieved")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in retrieval test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_ollama_llm_only(config: BackendConfig):
    """Test only the Ollama LLM connection without retrieval."""
    print(f"\n{'='*60}")
    print("TESTING OLLAMA LLM CONNECTION ONLY")
    print(f"{'='*60}")
    
    try:
        import time
        
        # Initialize LLM
        llm = LLM(
            model="ollama/llama3.2:1b",
            api_base=config.ollama_base_url,
            temperature=config.temperature,
            max_tokens=512,
            timeout=30,
            max_retries=2
        )

        print(f"🤖 Testing LLM: llama3.2:1b")
        print(f"🔗 Ollama URL: {config.ollama_base_url}")
        
        # Test simple completion
        test_prompt = "What are data types? Answer in 2 sentences."
        
        print(f"💬 Test prompt: '{test_prompt}'")
        print("⏳ Calling LLM...")
        
        start_time = time.time()
        
        # Use CrewAI's LLM call method
        response = llm.call([{"role": "user", "content": test_prompt}])
        
        end_time = time.time()
        
        print(f"⏱️ LLM call took: {end_time - start_time:.2f} seconds")
        print(f"✅ LLM Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in LLM test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_rag_crew(config: BackendConfig, query: str = "What data types are mentioned?"):
    """Test the full RAG CrewAI system."""
    print(f"\n{'='*60}")
    print("TESTING FULL RAG CREWAI SYSTEM")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print("-" * 40)
    
    try:
        import time
        
        # Mock RAG service for testing
        class MockRAGService:
            def __init__(self, config):
                self.config = config
                # Mock index for sources retrieval
                try:
                    DATABASE_URL = config.database_url
                    db_url_parts = urlparse(DATABASE_URL)
                    
                    vector_store = PGVectorStore.from_params(
                        host=db_url_parts.hostname,
                        port=db_url_parts.port,
                        database=db_url_parts.path.lstrip('/'),
                        user=db_url_parts.username,
                        password=db_url_parts.password,
                        table_name="llamaindex_vectors_copy",
                        embed_dim=768,
                    )
                    
                    embed_model = OllamaEmbedding(
                        model_name=config.ollama_embedding_model,
                        base_url=config.ollama_base_url,
                    )
                    
                    self.index = VectorStoreIndex.from_vector_store(
                        vector_store=vector_store,
                        embed_model=embed_model
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize index for sources: {e}")
                    self.index = None
        
        rag_service = MockRAGService(config)
        rag_crew = RAGCrew(rag_service, config)
        
        print("✅ RAG Crew initialized")
        
        start_time = time.time()
        
        result = await rag_crew.process_query(query)
        
        end_time = time.time()
        
        print(f"⏱️ Full processing took: {end_time - start_time:.2f} seconds")
        print(f"📝 Response: {result['response']}")
        print(f"📊 Metadata: {result['metadata']}")
        print(f"🔗 Sources: {len(result['sources'])} found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in full RAG test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests in sequence."""
    print("🧪 STARTING RAG CREWAI COMPREHENSIVE TESTS")
    print("=" * 80)
    
    # Initialize config - UPDATE THESE WITH YOUR ACTUAL VALUES
    config = BackendConfig()
    
    print(f"Configuration:")
    print(f"  Database: {config.database_url}")
    print(f"  Ollama URL: {config.ollama_base_url}")
    print(f"  LLM Model: llama3.2:1b")
    print(f"  Embedding Model: {config.ollama_embedding_model}")
    
    # Test 1: Document Retrieval Tool Only
    retrieval_success = test_retrieval_tool_only(config, "What is the primary purpose of the “Negotiation Plan” document?")
    
    # Test 2: LLM Only
    llm_success = test_ollama_llm_only(config)
    
    # Test 3: Full RAG Crew (only if previous tests pass)
    if retrieval_success and llm_success:
        print("\n✅ Basic tests passed, testing full RAG Crew...")
        try:
            import asyncio
            asyncio.run(test_full_rag_crew(config, "What is the primary purpose of the “Negotiation Plan” document?"))
        except Exception as e:
            print(f"❌ Full RAG Crew test failed: {e}")
    else:
        print("\n❌ Skipping full RAG test due to basic test failures")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"📄 Document Retrieval: {'✅ PASS' if retrieval_success else '❌ FAIL'}")
    print(f"🤖 LLM Connection: {'✅ PASS' if llm_success else '❌ FAIL'}")
    print(f"🔄 Integration Ready: {'✅ YES' if retrieval_success and llm_success else '❌ NO'}")


if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run comprehensive tests
    run_all_tests()
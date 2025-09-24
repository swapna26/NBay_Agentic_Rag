"""
CrewAI Agents for Intelligent RAG Processing

This module implements a multi-agent system using CrewAI for intelligent document
retrieval and response generation. The agents work collaboratively to provide
comprehensive and accurate answers to user queries.

Agent Architecture:
1. Document Retrieval Agent - Specialized in finding relevant documents
2. Analysis Agent - Analyzes and processes retrieved information
3. Response Generation Agent - Creates comprehensive responses


Version: 1.0.0
"""

import os
from typing import Dict, Any, List
import structlog
from urllib.parse import urlparse
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from config import BackendConfig

# Explicitly disable OpenAI for CrewAI to prevent API key errors
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''

logger = structlog.get_logger()


class RAGCrew:
    """
    CrewAI Multi-Agent System for Intelligent RAG Processing
    
    This class orchestrates a team of specialized AI agents that work collaboratively
    to provide comprehensive and accurate responses to user queries. Each agent has
    a specific role in the document retrieval and response generation process.
    
    Agent Team:
    1. Document Retrieval Agent - Finds relevant documents from vector store
    2. Analysis Agent - Analyzes and processes retrieved information
    3. Response Generation Agent - Creates comprehensive, well-structured responses
    
    Features:
    - Multi-agent collaboration with specialized roles
    - PostgreSQL vector store integration
    - Ollama LLM integration for local processing
    - Comprehensive error handling and logging
    """

    def __init__(self, rag_service, config):
        """
        Initialize the RAG Crew with specialized agents.
        
        Args:
            rag_service: Reference to the main RAG service
            config: Configuration object with database and model settings
        """
        self.rag_service = rag_service
        self.config = config
        self.crew = None

        # Configure Ollama LLM for CrewAI agents - Optimized for llama3.2:1b
        self.llm = LLM(
            model=f"ollama/{self.config.ollama_model}",
            api_base=self.config.ollama_base_url,
            temperature=0.0,  # Lower temperature for faster, more deterministic responses
            max_tokens=1200,  # Increased to allow more detailed responses
            timeout=90,       # Increased timeout for better response generation
            max_retries=1     # Keep at 1 for faster failure handling
        )

        # Initialize the agent team
        self._initialize_agents()

    def _create_document_retrieval_tool(self):
        """Create a document retrieval tool using the @tool decorator."""

        @tool("Search Documents")
        def search_documents(search_terms: str) -> str:
            """Search for relevant documents using provided search terms.

            Args:
                search_terms: Simple search keywords as a string
            """
            try:
                logger.info("Document search called", search_terms=search_terms[:100])

                # Validate we have a proper query string
                if not search_terms or not search_terms.strip():
                    return "Error: Search query cannot be empty."

                search_query = search_terms.strip()

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
        
        return search_documents

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

        # Enhanced follow-up question removal patterns
        follow_up_patterns = [
            r"^\s*(Would you like|Do you want|Can I help|Is there anything else).*?\?.*$",
            r"^\s*(Feel free to ask|Please let me know|If you have|Any other questions).*$",
            r"^\s*(I hope this helps|Hope this answers|Let me know if).*$",
            r"^\s*(For more information|To learn more|Additional details).*$",
            r"^\s*(Next steps|What would you like|How can I assist).*$",
            r"^\s*Would you like me to.*?\?.*$",
            r"^\s*Is there anything specific.*?\?.*$",
            r"^\s*Do you need.*?\?.*$",
            r"^\s*Can I provide.*?\?.*$",
        ]

        for pattern in follow_up_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # Remove FOLLOW_UP/NEW_TOPIC markers that might leak into responses
        cleaned = re.sub(r"(FOLLOW_UP|NEW_TOPIC)\s*[-:]\s*", "", cleaned, flags=re.IGNORECASE)

        # Remove meta-commentary phrases
        meta_phrases = [
            r"^\s*(Based on the conversation|According to the context|From the retrieved documents).*?(?=\n|$)",
            r"^\s*(The document shows|The information indicates|As mentioned).*?(?=\n|$)",
            r"^\s*(I can see that|It appears that|The response shows).*?(?=\n|$)",
        ]

        for pattern in meta_phrases:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # Remove excessive bold formatting - convert **text** to plain text
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)

        # Remove any remaining asterisks used for emphasis
        cleaned = re.sub(r'\*([^*\n]+)\*', r'\1', cleaned)

        # Strip all Markdown headings (avoid bold rendering in OpenWebUI)
        cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)

        # Strip common list markers to keep plain text (no bullets)
        cleaned = re.sub(r'^\s*[-*]\s+', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)

        # Remove prompt-leak style lines and validator/planner artifacts
        cleaned = re.sub(r'^\s*Your\s+final\s+answer\s+must.*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'^\s*Current\s*Task:.*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'^\s*Review\s+the\s+draft\s+response.*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'^\s*(Questions|Next\s*steps)\s*:.*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'^\s*I\s+(now\s+)?can\s+(give|provide).*$','', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'^\s*(Here\s+is|Let\s+me)\b.*$','', cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # Remove embedded "Sources" section from the model output (we return sources separately)
        cleaned = re.sub(r'^\s*#?\s*Sources:?\s*$[\s\S]*', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

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
        
        # Document Retrieval Agent (context-aware)
        self.retrieval_agent = Agent(
            role="Context-Aware Document Finder",
            goal="Use conversation context to find the most relevant documents",
            backstory="""You are a smart document finder who understands conversation context. Your job:

            1. ANALYZE THE FULL QUERY: Look for conversation context and current question
            2. UNDERSTAND FOLLOW-UPS: If there's previous context, understand what the user is referring to
            3. SEARCH APPROPRIATELY:
               - For new questions: Search using the question keywords
               - For follow-ups: Search using BOTH context keywords AND current request
            4. Use the Search Documents tool ONCE with the best keywords

            CRITICAL RULES:
            - Call the Search Documents tool only ONE time
            - For follow-up questions like "list in bullet points", use keywords from PREVIOUS context
            - Include specific terms mentioned in quotes or technical terms
            - Do NOT call the tool multiple times

            Examples:
            - "What is procurement?" → Search "procurement definition processes"
            - "Can you list above in bullet points?" (with context about disciplinary actions) → Search "disciplinary actions financial rules violations penalty clauses"
            - "How do X and Y relate?" → Search "X Y relationship processes"

            ALWAYS examine the full query for conversation context before choosing search terms.""",
            tools=[retrieval_tool],
            llm=self.llm,
            verbose=False,  # Turn off verbose to reduce confusion
            allow_delegation=False,
            max_iter=1,     # Only 1 iteration to prevent loops
            max_execution_time=120  # Give enough time for tool execution
        )
        
        # Query Analyzer functionality merged into Document Retrieval Agent above
        
        # Response Generation Agent (context-aware)
        self.response_agent = Agent(
            role="Context-Aware Answer Writer",
            goal="Write answers that understand conversation context and follow-up requests",
            backstory="""You are a smart answer writer who understands conversation flow. Your job:

            1. ANALYZE THE QUERY: Check if there's conversation context provided
            2. UNDERSTAND THE REQUEST:
               - If it's a new question: Provide a comprehensive answer
               - If it's a follow-up: Transform the PREVIOUS answer according to the current request
            3. READ DOCUMENTS: Use the retrieved documents to provide accurate information

            SPECIAL HANDLING FOR FOLLOW-UPS:
            - "List in bullet points" → Convert previous content to bullet format
            - "Summarize in X points" → Create X key points from previous answer
            - "What about Y?" → Focus on Y aspect from previous context
            - "Can you explain more?" → Expand on previous answer

            RULES:
            - Write in plain text only (no formatting symbols)
            - For bullet points, use simple lines without • or - symbols
            - Be comprehensive and accurate
            - Use information from documents
            - Don't ask follow-up questions
            - Don't say "Here is" or "Let me" - just answer directly""",
            llm=self.llm,
            verbose=False,  # Turn off verbose
            allow_delegation=False,
            max_iter=1,     # Only 1 iteration
            max_execution_time=90  # Enough time for detailed response generation
        )
        
        # Quality Validation Agent
        self.validation_agent = Agent(
            role="Response Formatter",
            goal="Format responses properly and ensure they answer the user's question",
            backstory="""Ensure responses:
            - Answer only what the user asked
            - Contain NO follow-up questions, NO suggested next steps
            - Are in plain text (no Markdown, no headings, no bullets)
            - Are concise and specific
            """,
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=1,        # Keep at 1 iteration
            max_execution_time=60  # Reduced from 180s to 60s
        )
    
    def create_crew(self, query: str) -> Crew:
        """Create a crew for processing a specific query."""

        # Streamlined workflow with 2 agents only

        # Task 1: Very simple document retrieval
        retrieval_task = Task(
            description=f"""Find documents for: {query}

Use the Search Documents tool ONCE with good keywords. Don't overthink it.""",
            agent=self.retrieval_agent,
            expected_output="Documents found using the search tool"
        )
        
        # Task 2: Comprehensive response generation
        response_task = Task(
            description=f"""Answer this question thoroughly using the documents: {query}

Provide a comprehensive answer that:
1. Addresses all parts of the question
2. Includes specific details from the documents
3. Explains relationships and processes clearly
4. Uses plain text format only

Be thorough and detailed in your response.""",
            agent=self.response_agent,
            expected_output="Comprehensive detailed answer in plain text",
            context=[retrieval_task]
        )

        # Create simple crew - no complications
        crew = Crew(
            agents=[self.retrieval_agent, self.response_agent],
            tasks=[retrieval_task, response_task],
            process=Process.sequential,
            verbose=False,  # Turn off all verbose output
            memory=False,   # No memory complications
            max_execution_time=210  # Total time for both agents
        )
        
        return crew
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query using the multi-agent CrewAI system.
        
        This method orchestrates the complete workflow:
        1. Creates a specialized crew for the query
        2. Executes the multi-agent workflow
        3. Extracts and cleans the final response
        4. Retrieves relevant source documents
        5. Returns structured response with metadata
        
        Args:
            query (str): User's question or query
            
        Returns:
            Dict[str, Any]: Structured response containing:
                - response: Generated answer
                - sources: List of source documents
                - metadata: Processing information
                
        Raises:
            Exception: If processing fails
        """
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
                    "agents_used": ["intelligent_retrieval_specialist", "information_extractor"],  # Only 2 agents now
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
        
        print(" Vector store connection successful")
        
        # Initialize embedding model
        embed_model = OllamaEmbedding(
            model_name=config.ollama_embedding_model,
            base_url=config.ollama_base_url,
        )
        
        print(" Embedding model initialized")
        
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        
        print(" Index created")
        
        # Test retrieval
        retriever = index.as_retriever(
            similarity_top_k=config.similarity_top_k,
            verbose=True
        )
        
        print(f" Retrieving documents for: '{query}'")
        start_time = time.time()
        
        nodes = retriever.retrieve(query)
        
        end_time = time.time()
        print(f" Retrieval took: {end_time - start_time:.2f} seconds")
        print(f" Found {len(nodes)} documents")
        
        if nodes:
            print("\n RETRIEVED DOCUMENTS:")
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
            print(" No documents retrieved")
            
        return True
        
    except Exception as e:
        print(f" Error in retrieval test: {str(e)}")
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

        print(f" Testing LLM: llama3.2:1b")
        print(f"🔗 Ollama URL: {config.ollama_base_url}")
        
        # Test simple completion
        test_prompt = "What are data types? Answer in 2 sentences."
        
        print(f"💬 Test prompt: '{test_prompt}'")
        print("⏳ Calling LLM...")
        
        start_time = time.time()
        
        # Use CrewAI's LLM call method
        response = llm.call([{"role": "user", "content": test_prompt}])
        
        end_time = time.time()
        
        print(f" LLM call took: {end_time - start_time:.2f} seconds")
        print(f" LLM Response: {response}")
        
        return True
        
    except Exception as e:
        print(f" Error in LLM test: {str(e)}")
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
        
        print(" RAG Crew initialized")
        
        start_time = time.time()
        
        result = await rag_crew.process_query(query)
        
        end_time = time.time()
        
        print(f" Full processing took: {end_time - start_time:.2f} seconds")
        print(f"📝 Response: {result['response']}")
        print(f" Metadata: {result['metadata']}")
        print(f"🔗 Sources: {len(result['sources'])} found")
        
        return True
        
    except Exception as e:
        print(f" Error in full RAG test: {str(e)}")
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
        print("\n Basic tests passed, testing full RAG Crew...")
        try:
            import asyncio
            asyncio.run(test_full_rag_crew(config, "What is the primary purpose of the “Negotiation Plan” document?"))
        except Exception as e:
            print(f" Full RAG Crew test failed: {e}")
    else:
        print("\n Skipping full RAG test due to basic test failures")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f" Document Retrieval: {' PASS' if retrieval_success else ' FAIL'}")
    print(f" LLM Connection: {' PASS' if llm_success else ' FAIL'}")
    print(f" Integration Ready: {' YES' if retrieval_success and llm_success else ' NO'}")


if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run comprehensive tests
    run_all_tests()
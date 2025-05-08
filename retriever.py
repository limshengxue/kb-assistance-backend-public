from grpc import ServicerContext
from llama_cloud import TextNode
from ingest import DocumentParser
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.mistralai import MistralAIEmbedding
import os

embedding_model_name = "mistral-embed"
collection_name = "digicelcom_documents"

class Retriever:
    def __init__(self):
        # Initialize ChromaDB client
        self.db = chromadb.PersistentClient(path="./chroma_db")
    
    async def initialize(self):
        # Check if collection exists and create index accordingly
        if self._collection_exists():
            self.index = self.load_index()
            print(f"Collection '{collection_name}' already exists. Loaded existing index.")
        else:
            self.index = await self.build_index()
            print(f"Collection '{collection_name}' created and documents indexed.")
    
    def _collection_exists(self):
        """Check if the collection exists in the ChromaDB database."""
        try:
            # Try to get the collection - if it exists, this succeeds
            # If it doesn't exist, this raises an exception
            collections = self.db.list_collections()            
            return any(col == collection_name for col in collections)
        except Exception as e:
            print(f"Error checking collection existence: {str(e)}")
            return False
    
    def load_index(self):
        """Load the existing vector index from ChromaDB."""
        chroma_collection = self.db.get_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create embed model for queries
        embed_model = MistralAIEmbedding(api_key=os.environ.get("MISTRALAI_API_KEY"), model=embedding_model_name)

        
        # Load index from existing vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_model
        )
        return index

    async def build_index(self):
        """Build a new vector index from documents and store in ChromaDB."""
        documents = await self.build_documents()
        print(f"Number of documents before indexing: {len(documents)}")

        # Create a new collection
        chroma_collection = self.db.create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create embed model
        embed_model = MistralAIEmbedding(api_key=os.environ.get("MISTRALAI_API_KEY"), model=embedding_model_name)
        
        from llama_index.core.schema import TextNode
        # Manually convert documents to nodes (bypassing automatic chunking)
        nodes = []
        for doc in documents:
            node = TextNode(
                text=doc.get_content(),
                metadata=doc.metadata
            )
            nodes.append(node)
        
        print(f"Number of nodes created: {len(nodes)}")
        
        # Create index directly from nodes (skipping document processing)
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model
        )
        
        return index

    async def build_documents(self):
        """Parse PDF documents and convert to LlamaIndex documents."""
        parser = DocumentParser()
        file_names = [
            "documents\\celcomdigi_port-in-rebate-offer.pdf",
            "documents\\celcomdigi_raya_video_internet_pass.pdf",
            "documents\\celcomdigi_samsung_galaxy_s25_series_launch.pdf",
            "documents\\celcomdigi-eratkanikatan-sahur-moreh-pass.pdf",
        ]
        documents = []

        for file in file_names:
            print(f"Processing document: {file}")
            try:
                markdown_result = await parser.parse(file)
                document = Document(
                    text=markdown_result,
                    metadata={"filename": file}
                )
                documents.append(document)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
        
        print(f"Successfully processed {len(documents)} documents")
        return documents

    def get_retriever(self):
        return self.index.as_retriever(similarity_top_k=4, vector_store_query_mode = "hybrid")

    def get_index(self):
        return self.index
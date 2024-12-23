from src.helper import load_pdf_file, split_text, download_embeddings_model
from pinecone.grpc import PineconeGRPC
from pinecone import ServerlessSpec
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

#load PDF file
extracted_data = load_pdf_file(data='data/')
chunks = split_text(extracted_data)
embeddings = download_embeddings_model()

pc = PineconeGRPC(api_key=PINECONE_API_KEY)

index_name = "telemedai"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 


docsearch = LangchainPinecone.from_documents(
    documents=chunks,
    embedding=embeddings,  # Changed from embeddings to embedding
    index_name=index_name,
)
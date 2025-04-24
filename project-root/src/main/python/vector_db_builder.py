# vector_builder.py

from pathlib import Path
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


# -------------------------------------
def get_file_paths():
    """
    Returns file paths for the input CSV and the output vector DB.

    Returns:
        tuple: A tuple containing:
            - file_path (Path): Path to the CSV file.
            - vector_db_path (Path): Path to save the FAISS vector database.
    """
    print("📁 Getting file paths...")
    base_path = Path(__file__).resolve().parents[3]
    file_path = base_path / "data" / "car_details_1.csv"
    vector_db_path = base_path / "data" / "faiss_index"
    print(f"✅ CSV Path: {file_path}")
    print(f"✅ Vector DB Path: {vector_db_path}")
    return file_path, vector_db_path

# -------------------------------------
def load_csv_as_documents(file_path):
    """
    Loads a CSV file and converts its rows into LangChain document objects.

    Args:
        file_path (Path): The path to the CSV file.

    Returns:
        List[Document]: A list of LangChain documents, one for each row in the CSV.
    """
    print("📦 Loading CSV file into documents...")
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents.")
    return documents

# -------------------------------------
def create_vector_db(documents):
    """
    Creates a FAISS vector database from a list of documents.

    Args:
        documents (List[Document]): The documents to embed and store.

    Returns:
        FAISS: A FAISS vector database containing the embedded documents.
    """
    print("🔧 Creating vector database...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(documents)
    print(f"🧩 Split into {len(split_docs)} chunks.")
    db = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    print("✅ Vector database created.")
    return db

# -------------------------------------
def save_vector_db(db, vector_db_path):
    """
    Saves the FAISS vector database to a specified local directory.

    Args:
        db (FAISS): The FAISS database instance to save.
        vector_db_path (Path): The path where the vector database will be saved.

    Returns:
        None
    """
    print("💾 Saving vector database locally...")
    db.save_local(str(vector_db_path))
    print(f"✅ Vector DB saved at: {vector_db_path}")

# -------------------------------------
def main():
    """
    Main function to orchestrate the loading, processing, vectorization,
    and saving of CSV data into a FAISS vector database.

    Returns:
        None
    """
    print("🚀 Starting vector DB creation pipeline...\n")
    file_path, vector_db_path = get_file_paths()
    documents = load_csv_as_documents(file_path)
    db = create_vector_db(documents)
    save_vector_db(db, vector_db_path)
    print("\n🏁 All steps completed successfully!")

# -------------------------------------
if __name__ == "__main__":
    print("🔐 Loading environment variables...")
    dotenv_path = Path(__file__).resolve().parents[3] / ".env"
    load_dotenv(dotenv_path)
    print("✅ Environment variables loaded.\n")
    main()











# garbage code below - 



# # print("Hello")

# # This is the code to take the csv file and then make the vector db which can be used for rag 


# ###############################
# # this part loads the csv file  

# from langchain_community.document_loaders.csv_loader import CSVLoader

# from pathlib import Path
# import pandas as pd

# # Dynamically get the base path — adjust how many parents based on your file location
# base_path = Path(__file__).resolve().parents[3]  # goes up 3 levels from the current file
# file_path = base_path / "data" / "car_details_1.csv"


# loader = CSVLoader(file_path=file_path)
# documents = loader.load()

# ############################################################
# # making the vector db 

# import os
# import getpass

# os.environ['OPENAI_API_KEY'] = """

# from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import CharacterTextSplitter

# # Load the document, split it into chunks, embed each chunk and load it into the vector store.
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# split_docs = text_splitter.split_documents(documents)

# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OpenAIEmbeddings


# db = FAISS.from_documents(split_docs, OpenAIEmbeddings())

# ###################################################

# # db.save_local("faiss_index")
# # Define path to store vector DB
# vector_db_path = base_path / "data" / "faiss_index"
# db.save_local(str(vector_db_path))

# #####################################################


# # query = "Show me a petrol car under 5 lakh"
# # docs = db.similarity_search(query)
# # print(docs[0].page_content)


# # #############################################
# # # TOOL CALLING SECTION: Google Search Tool
# # #############################################

# # from langchain.tools import tool
# # from langchain.agents import initialize_agent
# # from langchain.agents.agent_types import AgentType
# # from langchain.chat_models import ChatOpenAI

# # # Dummy tool that simulates Google search
# # @tool
# # def search_google(query: str) -> str:
# #     """Use Google search to get more info about a car."""
# #     return f"Simulated Google results for: '{query}'"

# # # Initialize the tool + agent
# # tools = [search_google]
# # llm = ChatOpenAI(temperature=0)

# # agent = initialize_agent(
# #     tools=tools,
# #     llm=llm,
# #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# #     verbose=True,
# # )

# # # Get top vector search result from earlier
# # car_name = docs[0].page_content

# # # Call the tool via agent
# # response = agent.run(f"Search the web for more info about this car: {car_name}")

# # print("\n🔎 Tool Result:")
# # print(response)

# #########################################################################



# # from langchain_community.vectorstores import FAISS
# # from langchain_openai import OpenAIEmbeddings

# # # db = FAISS.load_local("faiss_index", OpenAIEmbeddings())
# # # Same base path logic
# # base_path = Path(__file__).resolve().parents[3]
# # vector_db_path = base_path / "data" / "faiss_index"

# # # Load the vector DB
# # db = FAISS.load_local(str(vector_db_path), OpenAIEmbeddings())
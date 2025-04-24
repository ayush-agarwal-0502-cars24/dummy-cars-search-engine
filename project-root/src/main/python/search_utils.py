from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from tavily import TavilyClient
from langchain.memory import ConversationBufferMemory


def load_vector_db() -> FAISS:
    """
    Load the FAISS vector database from local storage.

    Returns:
        FAISS: The loaded vector database.
    """
    base_path = Path(__file__).resolve().parents[3]
    vector_db_path = base_path / "data" / "faiss_index"
    db = FAISS.load_local(
        str(vector_db_path),
        OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )
    return db


def get_user_query() -> str:
    """
    Get the user's query for searching the car database.

    Returns:
        str: The query string (currently hardcoded for demo).
    """
    return "Show me a petrol car under 5 lakh"  # Can be replaced with input() for dynamic queries


def retrieve_relevant_docs(db: FAISS, query: str, top_k: int = 5) -> str:
    """
    Retrieve the top K most similar documents from the vector database.

    Args:
        db (FAISS): The loaded vector database.
        query (str): User's input query.
        top_k (int): Number of top matches to retrieve.

    Returns:
        str: Combined page content of the top matches.
    """
    docs = db.similarity_search(query, k=top_k)
    print("\n📄 Top Vector DB Matches:\n")
    for i, doc in enumerate(docs, 1):
        print(f"{i}. {doc.page_content[:100]}...\n")
    combined_content = "\n\n".join([doc.page_content for doc in docs])
    return combined_content


@tool
def search_google(query: str) -> str:
    """
    Simulated Google search tool for testing.

    Args:
        query (str): Search query.

    Returns:
        str: Simulated search result.
    """
    return f"Simulated Google results for: '{query}'"


@tool
def search_web(query: str) -> str:
    """
    Search the web using the Tavily API and return top snippets.

    Args:
        query (str): The search query.

    Returns:
        str: Concatenated content from top Tavily results.
    """
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query=query, search_depth="basic", max_results=3)
    snippets = [res["content"] for res in results["results"]]
    return "\n".join(snippets)

# this was older code of agent without memory 
# def setup_agent():
#     """
#     Set up a LangChain agent with tool-calling capability.

#     Returns:
#         AgentExecutor: Initialized agent with tools and LLM.
#     """
#     tools = [search_web]
#     llm = ChatOpenAI(temperature=0)
#     agent = initialize_agent(
#         tools=tools,
#         llm=llm,
#         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#         verbose=True,
#     )
#     return agent


def setup_agent():
    tools = [search_web]
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True
    )
    return agent

def get_ai_summary(agent, car_description: str, user_query: str) -> str:
    """
    Uses the agent to generate a detailed AI summary about the car, its pros/cons,
    and how it matches the user's query.
    """
    prompt = (
        f"Given the following car info:\n\n{car_description}\n\n"
        f"Describe the car in detail, including:\n"
        f"- Overview of the car (company, model, type)\n"
        f"- Key features, pros and cons\n"
        f"- Why it fits (or doesn't fit) the user query: '{user_query}'\n"
    )
    return agent.run(prompt)

def main():
    """
    Main execution function to perform vector search and invoke tool-calling agent.
    """
    load_dotenv()
    db = load_vector_db()
    query = get_user_query()
    car_info = retrieve_relevant_docs(db, query)
    agent = setup_agent()
    # response = agent.run(f"Search the web for more info about this car: {car_info}")
    response = agent.run(
        f"""
        Based on the following information, give a comprehensive recommendation for the user:
        
        1. Describe the company selling the car (if mentioned).
        2. Describe the car model.
        3. List its pros and cons clearly.
        4. Explain why this car is a good fit for the user's query: '{query}'.

        Car info:\n{car_info}
        """
    )

    print("\n🔎 Tool Result:\n", response)


if __name__ == "__main__":
    main()
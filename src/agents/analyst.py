import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from src.tools.retriever import SportsRetriever

load_dotenv(find_dotenv())

# --- 1. Define the Tools ---

@tool
def search_knowledge_base(query: str, sport: str = None):
    """
    Search the sports database for scores, stats, and news.
    Always use this tool to answer questions about games, teams, or players.
    
    Args:
        query: The user's question (e.g. "Who won the Patriots game?").
        sport: (Optional) 'nba' or 'nfl'.
    """
    retriever = SportsRetriever()
    results = retriever.search(query, sport=sport, top_k=4)
    
    context = ""
    for i, res in enumerate(results):
        context += f"\n[Result {i+1}] {res['text']}"
        if res.get('context'):
            context += f"\nContext: {res['context']}"
            
    return context if context else "No results found in the database."

# --- 2. Define the Agent Factory ---

def create_analyst_agent():
    """
    Constructs the Sports Analyst Agent using LangGraph.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = [search_knowledge_base]
    
    # SIMPLE FIX: We do not pass 'state_modifier' or 'messages_modifier' here.
    # We will inject the System Prompt when we INVOKE the agent.
    graph = create_react_agent(llm, tools)
    return graph

# --- 3. Interactive Test Loop ---

if __name__ == "__main__":
    print("🤖 Sports Analyst Agent Online (Universal Mode)...")
    
    agent_graph = create_analyst_agent()
    
    # Define the Persona HERE
    system_prompt = SystemMessage(content=(
        "You are an expert Sports Analyst backed by a real-time database.\n"
        "Your goal is to provide accurate, data-driven summaries of games and news.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. ALWAYS use the 'search_knowledge_base' tool first. Do not guess scores.\n"
        "2. If a game is 'Scheduled' or 'Preview', explicitly state that it hasn't happened yet.\n"
        "3. Use the 'Context' field (Odds, Weather) to add color to your commentary."
    ))
    
    config = {"configurable": {}}
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            # We pass the System Prompt as the FIRST message every time (or maintain history)
            # For this test loop, we just prepend it to the current turn.
            messages = [system_prompt, HumanMessage(content=user_input)]
            
            response = agent_graph.invoke({"messages": messages}, config)
            
            # Extract the final AI message
            print(f"\nAnalyst: {response['messages'][-1].content}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
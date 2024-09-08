import os
import platform
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_community.tools.google_finance import GoogleFinanceQueryRun
from langchain_community.utilities.google_finance import GoogleFinanceAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

# Function to save the API key to a .env file without overwriting other keys
def save_api_key_to_env(key_name, api_key):
    # Check if the .env file already contains the key
    if not os.path.exists(".env"):
        with open(".env", "w") as env_file:
            env_file.write(f"{key_name}={api_key}\n")
    else:
        # If the file exists, ensure you are not duplicating keys
        with open(".env", "r") as env_file:
            lines = env_file.readlines()

        # Check if key already exists and modify it
        updated = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key_name}="):
                lines[i] = f"{key_name}={api_key}\n"
                updated = True
        
        if not updated:
            lines.append(f"{key_name}={api_key}\n")

        # Write back to the file
        with open(".env", "w") as env_file:
            env_file.writelines(lines)

    print(f"{key_name} saved to .env file.")

# Function to get and verify API keys
def get_api_key(key_name):
    # Check if API key is present in environment variables
    api_key = os.getenv(key_name)
    if not api_key:
        # Prompt the user to enter the API key if not found
        manual_api_key = input(f"Enter your {key_name}: ")
        save_api_key_to_env(key_name, manual_api_key)
        # Reload the environment with the new API key
        load_dotenv()
        api_key = os.getenv(key_name)
    return api_key

# Load environment variables from the .env file
load_dotenv()

# Get the environment variables for the API keys
anthropic_api_key = get_api_key("ANTHROPIC_API_KEY")
serp_api_key = get_api_key("SERPAPI_API_KEY")

# Tools
g_tool = GoogleFinanceQueryRun(api_wrapper=GoogleFinanceAPIWrapper(serp_api_key=serp_api_key))
y_tool = YahooFinanceNewsTool()

# Create the agent
memory = MemorySaver()
model = ChatAnthropic(model_name="claude-3-sonnet-20240229", api_key=anthropic_api_key)

# Combine tools
tools = [g_tool, y_tool]

agent_executor = create_react_agent(model, tools, checkpointer=memory)

# Configuration for the agent
config = {"configurable": {"thread_id": "abc123"}}

# Get the OS name
os_name = platform.system()

# Use the agent in a loop
while True:
    prompt = input("Enter a prompt: ")

    # Use the agent to handle the user prompt
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=prompt)]}, config
    ):
        print(chunk)
        print("----")

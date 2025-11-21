import operator
from typing import Annotated, List, Literal, Union
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import MemorySaver

# --- 1. DEFINE STRUCTURED OUTPUT (Link 1) ---
class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    content: str

class TableBlock(BaseModel):
    type: Literal["table"] = "table"
    headers: List[str]
    rows: List[List[str]]

class StepOutput(BaseModel):
    """
    The container for a SINGLE block. 
    The agent must use this to return one block at a time.
    """
    block: Union[TextBlock, TableBlock] = Field(..., description="The single content block.")
    is_finished: bool = Field(
        ..., description="True if the report is fully complete, False if more blocks are needed."
    )

# --- 2. CREATE AGENT (Link 2) ---
# We use 'create_agent' which is the standard in v1.0.
# We uses 'response_format' with 'ToolStrategy' to force the structured output.

model = ChatOpenAI(model="gpt-4o", temperature=0)
checkpointer = MemorySaver() # Logic for Short-Term Memory (Link 3)

system_prompt = (
    "You are a reporting assistant. "
    "Generate the report ONE BLOCK at a time. "
    "1. Look at the history to see what you have already generated. "
    "2. Generate the next logical block (Introduction -> Data -> Conclusion). "
    "3. If you have more to add, set is_finished=False. "
    "4. If done, set is_finished=True."
)

agent = create_agent(
    model=model,
    tools=[], # No external tools needed, we just want the structured response
    system_prompt=system_prompt,
    checkpointer=checkpointer,
    response_format=ToolStrategy(StepOutput)
)

# --- 3. EXECUTION LOOP ---
# The agent processes one step, returns the structure, and pauses. 
# You control the "next iteration" via this loop.

config = {"configurable": {"thread_id": "v1_session_1"}}
user_input = "Report on Apple 2023 with a summary and a table."

# We send the initial message
current_request = {"messages": [("user", user_input)]}

print("--- STARTING v1 AGENT LOOP ---")

while True:
    # Invoke the agent. 
    # Because we used 'response_format', it runs until it produces 'StepOutput'.
    result = agent.invoke(current_request, config=config)
    
    # Extract the Pydantic object (Structured Output)
    response: StepOutput = result["structured_response"]
    
    # --- PROCESS THE BLOCK ---
    print(f"\n📦 Generated Block: {response.block.type.upper()}")
    
    if response.block.type == "text":
        print(f"Content: {response.block.content}")
    elif response.block.type == "table":
        print(f"Table: {response.block.headers}")
        print(f"Rows: {len(response.block.rows)}")

    # --- CHECK COMPLETION ---
    if response.is_finished:
        print("\n🏁 Agent signaled completion.")
        break

    # --- NEXT ITERATION ---
    # The agent has 'Short-Term Memory' via the checkpointer. 
    # We just trigger the next step.
    current_request = {"messages": [("user", "Proceed to the next block.")]}
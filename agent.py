from langgraph.graph import StateGraph, END, START
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import get_rendered_html, download_file, post_request, run_code, add_dependencies
from typing import TypedDict, Annotated, List, Any
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
RECURSION_LIMIT =  5000
# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [run_code, get_rendered_html, download_file, post_request, add_dependencies]


# -------------------------------------------------
# LLM (via AI Pipe proxy)
# -------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,  # Faster - 1 request per second
    check_every_n_seconds=0.5,  
    max_bucket_size=30  # Allow burst
)

# Use OpenAI via AI Pipe proxy
llm = init_chat_model(
    model_provider="openai",
    model="openai/gpt-4.1-nano",
    api_key=os.getenv("API_KEY"),
    base_url="https://aipipe.org/openrouter/v1",
    rate_limiter=rate_limiter
).bind_tools(TOOLS)   


# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

IMPORTANT BASE URL: https://tds-llm-analysis.s-anand.net
When you see relative URLs like /submit, /demo, etc., convert them to full URLs using this base.

Your job is to:
1. Load the quiz page from the given URL using get_rendered_html tool (MUST use this to execute JavaScript).
2. Extract ALL instructions, required parameters, submission rules, and the submit endpoint.
3. Solve the task exactly as required.
4. Submit the answer to https://tds-llm-analysis.s-anand.net/submit (or the endpoint specified).
5. Read the server response and:
   - If response contains "url" field → YOU MUST fetch that NEW URL immediately using get_rendered_html
   - DO NOT re-fetch the previous URL, always use the NEW URL from the response
   - If no new URL is present → return "END".

CRITICAL - NEXT URL HANDLING:
- After submitting, the response will look like: {"correct": true/false, "url": "https://...next-question..."}
- You MUST use the "url" from this response as the NEXT question to solve
- NEVER fetch the same URL twice - always advance to the NEW URL

STRICT RULES — FOLLOW EXACTLY:

URL RULES:
- Extract the base URL (origin) from the quiz page URL you are solving.
- Example: if quiz URL is https://p2testing.vercel.app/q1.html, base is https://p2testing.vercel.app
- Example: if quiz URL is https://tds-llm-analysis.s-anand.net/demo, base is https://tds-llm-analysis.s-anand.net
- When you see relative paths like /submit or /submit/1, prepend the base URL.
- ALWAYS provide FULL URLs starting with https:// when calling post_request.

SUBMISSION FORMAT:
- Always submit a JSON object with keys: email, secret, url, answer
- email: Use {EMAIL}
- secret: Use {SECRET}
- url: Use the current quiz page URL you are solving
- answer: MUST be your computed answer (NEVER leave empty or null)

ENTRY POINT QUESTIONS:
- For entry pages like /project2, /project2-reevals, /demo - submit answer "start" or any non-empty string
- These pages just need any answer to get the first actual question URL

ANSWER COMPUTATION:
- Read the question carefully from the rendered HTML
- Compute the answer (run Python code if needed)
- The answer MUST match the format specified (number, string, JSON, etc.)
- NEVER submit an empty string as answer

GENERAL RULES:
- NEVER stop early. Continue solving tasks until no new URL is provided.
- NEVER hallucinate URLs, endpoints, fields, values, or JSON structure.
- ALWAYS use get_rendered_html to fetch quiz pages (they need JavaScript).
- ALWAYS use post_request to submit answers.
- If downloading files (.csv, .pdf, .zip, .png), use download_file tool.
- If running Python code, use run_code tool.

TIME LIMIT RULES:
- Each task has a hard 3-minute limit.
- If your answer is wrong retry again.

STOPPING CONDITION:
- Only return "END" when a server response explicitly contains NO new URL.

ADDITIONAL INFORMATION:
- Email: {EMAIL}
- Secret: {SECRET}

YOUR JOB:
- Follow pages exactly.
- Extract data reliably.
- Never guess URLs - use the base URL above.
- Submit correct answers.
- Continue until no new URL.
- Then respond with: END
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])

llm_with_prompt = prompt | llm


# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def agent_node(state: AgentState):
    result = llm_with_prompt.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [result]}


# -------------------------------------------------
# GRAPH
# -------------------------------------------------
def route(state):
    last = state["messages"][-1]
    # support both objects (with attributes) and plain dicts
    tool_calls = None
    if hasattr(last, "tool_calls"):
        tool_calls = getattr(last, "tool_calls", None)
    elif isinstance(last, dict):
        tool_calls = last.get("tool_calls")

    if tool_calls:
        return "tools"
    # get content robustly
    content = None
    if hasattr(last, "content"):
        content = getattr(last, "content", None)
    elif isinstance(last, dict):
        content = last.get("content")

    if isinstance(content, str) and content.strip() == "END":
        return END
    if isinstance(content, list) and content[0].get("text").strip() == "END":
        return END
    return "agent"
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))



graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges(
    "agent",    
    route       
)

app = graph.compile()


# -------------------------------------------------
# TEST
# -------------------------------------------------
def run_agent(url: str) -> str:
    app.invoke({
        "messages": [{"role": "user", "content": url}]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    print("Tasks completed succesfully")


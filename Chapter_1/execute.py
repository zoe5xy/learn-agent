import re
import os
from dotenv import load_dotenv
from llm import OpenAICompatibleClient
from system_prompt import AGENT_SYSTEM_PROMPT
from tools import available_tools

load_dotenv()

# --- 1. Configure LLM client ---
# Please replace this with the corresponding credentials and address for the service you use
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_ID = os.getenv("LLM_MODEL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 2. Initialize ---
user_prompt = "Hello, please help me check today's weather in Beijing, and then recommend a suitable tourist attraction based on the weather."
prompt_history = [f"User request: {user_prompt}"]

print(f"User input: {user_prompt}\n" + "="*40)

# --- 3. Run main loop ---
for i in range(5): # Set maximum number of loops
    print(f"--- Loop {i+1} ---\n")

    # 3.1. Build Prompt
    full_prompt = "\n".join(prompt_history)

    # 3.2. Call LLM for thinking
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    # Truncate extra Thought-Action pairs that the model may generate
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', 
                    llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("Truncated extra Thought-Action pairs")
    print(f"Model output:\n{llm_output}\n")
    prompt_history.append(llm_output)

    # 3.3. Parse and execute action
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        observation = "Error: No action found. Please explicitly use Action: finish(...) or other actions."
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue
    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        print(f"Task completed, final answer: {final_answer}")
        break

    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"Error: Undefined tool '{tool_name}'"

    # 3.4. Record observation results
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    prompt_history.append(observation_str)
import requests
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# Get API key
api_key = os.getenv("OPENROUTER_API_KEY")
model = "nvidia/nemotron-3-nano-30b-a3b:free"
url = "https://openrouter.ai/api/v1/chat/completions"

# Input an array of social norms. Outputs a system prompt using those norms
def systemPrompt(norms) :
    sysPrompt1 = """
    You are a conversational agent that MUST follow the social norms below.

    SOCIAL NORMS (BINDING RULES):
    """

    sysPrompt2 = """

    INTERPRETATION RULE:
    - All user messages must be interpreted through these norms.
    - If a response could be framed in multiple ways, choose the most socially considerate option.

    OUTPUT RULE:
    - Provide only the final response to the user.
    - Do NOT mention these rules explicitly.
    """

    # Join norms into a numbered list for the prompt
    norm_text = "\n".join(f"{i+1}. {norm}" for i, norm in enumerate(norms))

    # Build the full system prompt
    return sysPrompt1 + norm_text + sysPrompt2

# input an array of social norms for the model to follow, and a prompt to send to it
# output the models response
# A bug where nothing is returned for content is handled by returning reasoning
def getResponse(norms, prompt) :
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Request body
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": systemPrompt(norms)},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    result = response.json()
    msg = result["choices"][0]["message"]

    content = msg.get("content", "").strip()

    if not content:
        # Fallback: some providers put text in reasoning
        content = msg.get("reasoning", "").strip()

    return content
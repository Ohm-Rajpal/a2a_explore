from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import hashlib
import re

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

load_dotenv()
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Agent Card
# -------------------------------------------------------------------

@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    return {
        "name": "General purpose intelligent agent",
        "description": (
            "An agent that can solve various problems: math, cryptography, "
            "image analysis, web browsing, code generation, and memory tasks"
        ),
        "capabilities": [
            "chat",
            "math",
            "crypto",
            "image analysis",
            "web browsing",
            "code gen",
            "task memorization"
        ],
        "version": "1.0"
    }

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

# Hashing
def compute_sha512(current: str) -> str:
    return hashlib.sha512(current.encode()).hexdigest()

def compute_md5(current: str) -> str:
    return hashlib.md5(current.encode()).hexdigest()

def compute_sequence(value: str, sequence: list[str]) -> str:
    """Apply sequence of hash operations to the input string."""
    ret_val = value
    for entry in sequence:
        entry = entry.lower()
        if entry == "sha512":
            ret_val = compute_sha512(ret_val)
        else:
            ret_val = compute_md5(ret_val)
    return ret_val

def extract_hash_sequence_and_input(text: str):
    """Extracts the input string and ordered hash operations from text."""
    input_string_match = re.search(r'"(.*?)"', text)
    input_string = input_string_match.group(1) if input_string_match else None
    operations = re.findall(r'\d+\.\s*(md5|sha512)', text, re.IGNORECASE)
    operations = [op.lower() for op in operations]
    return input_string, operations

# -------------------------------------------------------------------
# Capability Handlers
# -------------------------------------------------------------------

async def handle_math_or_qa(user_text: str, request_id: str):
    """Math/Q&A capability using OpenAI models."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a calculator. Solve the math problem and return ONLY the numerical answer."},
            {"role": "user", "content": user_text}
        ],
        max_tokens=10,
        temperature=0
    )
    answer = response.choices[0].message.content.strip()
    return format_a2a_response(answer, request_id)


async def handle_hashing(user_text: str, request_id: str):
    """Cryptography capability for hashing sequences."""
    input_string, operations = extract_hash_sequence_and_input(user_text.lower())
    computed_hash = compute_sequence(input_string, operations)
    return format_a2a_response(computed_hash, request_id)

# -------------------------------------------------------------------
# Core A2A Handler
# -------------------------------------------------------------------

@app.post("/")
async def handle_a2a_message(request: Request):
    try:
        body = await request.body()
        print(f"Raw request: {body.decode()}")

        data = await request.json()
        print(f"Parsed data: {json.dumps(data, indent=2)}")

        # Extract user text from A2A structure
        user_text = ""
        raw_image_bytes = None

        if "params" in data and "message" in data["params"]:
            message = data["params"]["message"]
            if "parts" in message:
                for part in message["parts"]:
                    if part.get("kind") == "text" and "text" in part:
                        user_text += part["text"]
                    elif part.get("kind") == "file":
                        raw_image_bytes = part["file"]["bytes"]

        print(f"Extracted text: '{user_text}'")

        # --- Dispatch to correct handler ---
        if "hash" in user_text.lower():
            print("Classified as hashing problem")
            return await handle_hashing(user_text, data.get("id", "1"))

        elif user_text:
            print("Classified as math/QA problem")
            return await handle_math_or_qa(user_text, data.get("id", "1"))

        # --- Default fallback ---
        return format_a2a_response("Ready for math!", data.get("id", "1"))

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            },
            "id": data.get("id", "1")
        }

# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

def format_a2a_response(text: str, request_id: str):
    """Format a standard A2A-compliant response."""
    result = {
        "jsonrpc": "2.0",
        "result": {
            "messageId": f"response-{request_id}",
            "role": "assistant",
            "parts": [{"kind": "text", "text": text}]
        },
        "id": request_id
    }
    print(f"Sending: {json.dumps(result, indent=2)}")
    return result

# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
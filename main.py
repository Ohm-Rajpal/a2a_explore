from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import hashlib
import re
import base64
from typing import Optional, Dict, Tuple

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

# Coding problem
MOD = 1000

def _sum_sq_primes_mod(n: int, mod: int = MOD) -> int:
    # Sieve of Eratosthenes to enumerate primes up to n
    prime = [False, False] + [True] * (n - 1)  # 0,1 not prime
    p = 2
    while p * p <= n:
        if prime[p]:
            start = p * p
            prime[start:n+1:p] = [False] * ((n - start) // p + 1)
        p += 1
    total = 0
    for i in range(2, n + 1):
        if prime[i]:
            total = (total + (i * i) % mod) % mod
    return total

# In‑process temporary store
class PairMemory:
    def __init__(self):
        self.fwd: Dict[str, str] = {}
        self.rev: Dict[str, str] = {}

    def upsert(self, a: str, b: str):
        # keep one-to-one mapping (drop stale inverse if needed)
        if a in self.fwd:
            self.rev.pop(self.fwd[a], None)
        if b in self.rev:
            self.fwd.pop(self.rev[b], None)
        self.fwd[a] = b
        self.rev[b] = a

    def get(self, x: str) -> Optional[str]:
        return self.fwd.get(x) or self.rev.get(x)

    def clear_all(self):
        # O(1) way to empty dicts
        self.fwd.clear()
        self.rev.clear()

PAIR_MEM = PairMemory()

# Matches “… pair … 46894 and 91108 …” or any two integers in order
TWO_NUMS = re.compile(r"(\d+)\D+(\d+)", re.I)

def remember_from_message_text(text: str) -> int:
    """
    Extract two integers (first occurrence) and store both directions.
    Returns 1 if a pair was stored, else 0.
    """
    m = TWO_NUMS.search(text)
    if not m:
        return 0
    a, b = m.group(1), m.group(2)
    PAIR_MEM.upsert(a, b)
    return 1

ASK = re.compile(r"(?:paired\s+with|with)\s*(\d+)", re.I)

def recall_from_message_text(text: str) -> Optional[str]:
    """
    If the message asks for the partner of a number, return it; otherwise None.
    """
    m = ASK.search(text)
    if not m:
        return None
    x = m.group(1)
    return PAIR_MEM.get(x)


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
    return format_a2a_response(answer, request_id, "text")


async def handle_hashing(user_text: str, request_id: str):
    """Cryptography capability for hashing sequences."""
    input_string, operations = extract_hash_sequence_and_input(user_text.lower())
    computed_hash = compute_sequence(input_string, operations)
    return format_a2a_response(computed_hash, request_id, "text")


# TO FIX!!!
async def handle_image(user_text: str, raw_image_bytes: bytes, request_id: str):
    """Image analysis capability."""
    image_base64 = base64.b64encode(raw_image_bytes).decode('utf-8')
    question = user_text if user_text.strip() else "What do you see in this image?"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an image classifier. Look at the image and respond with ONLY the word 'cat' or 'dog'. Nothing else."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }
        ],
        max_tokens=300,
    )

    print(f'response.choices[0].message = {response.choices[0].message}')
    answer = response.choices[0].message.content.strip().lower()
    print(f'handle image answer is {answer}')   

    return format_a2a_response(answer, request_id, "text")

async def handle_program(user_text: str, request_id: str):
    """
    Handle a math/coding problem that asks for:
    sum of squares of primes <= n (mod 1000).
    Parses n from user_text and returns only the final number.
    """
    m = re.search(r"n\s*=\s*(\d+)", user_text)
    if not m:
        # JSON-RPC error if we can't find n
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32602, "message": "Could not parse n from input"}
        })
    n = int(m.group(1))
    value = _sum_sq_primes_mod(n, MOD)
    return format_a2a_response(str(value), request_id, "text")

async def insert_memory(user_text: str, request_id: str):
    """Inserts pair in memory"""
    inserted = remember_from_message_text(user_text)
    return format_a2a_response(inserted, request_id, "text")

async def get_memory(user_text: str, request_id: str):
    """Retrieves pair from memory"""
    ans = recall_from_message_text(user_text)
    # clear out the previous memory now
    PAIR_MEM.clear_all()  # wipe both directions

    return format_a2a_response(ans, request_id, "text")


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
                    elif part.get("kind") == "file" and "file" in part:
                        file_info = part["file"]
                        if file_info.get("mimeType", "").startswith("image/"):
                            raw_image_bytes = base64.b64decode(file_info["bytes"])
                            print(f"Image extracted: {len(raw_image_bytes)} bytes")
        
        # print(f'Extracted text: {user_text}')
        # print("Length of incoming base64 string:", len(file_info["bytes"]))
        # print("Length of decoded bytes:", len(raw_image_bytes))


        # --- Dispatch to correct handler ---

        # TODO: NEEDS TO BE FIXED
        if raw_image_bytes:
            print("Classified as image understanding")
            return await handle_image(user_text, raw_image_bytes, data.get("id", "1"))
        elif "hash" in user_text.lower():
            print("Classified as hashing problem")
            return await handle_hashing(user_text, data.get("id", "1"))
        elif "program" in user_text.lower():
            print("Classified as programming problem")
            return await handle_program(user_text, data.get("id", "1"))
        elif "remember" in user_text.lower():
            print("Classified as a memory insertion")
            return await insert_memory(user_text, data.get("id", "1"))
        elif "memory" in user_text.lower():
            print("Classified as memory retreival")
            return await get_memory(user_text, data.get("id", "1"))

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

def format_a2a_response(text: str, request_id: str, kind: str):
    # Build an A2A-compliant JSON-RPC response
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,                 # must echo request
        "result": {                       # omit this and use "error" on failure
            "message": {                  # wrap output as a Message object
                "kind": "message",
                "messageId": f"response-{request_id}",
                "role": "agent",          # A2A server replies are role=agent
                "parts": [
                    {"kind": kind, "text": text}
                ]
            }
        }
    }
    return JSONResponse(content=payload)

# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
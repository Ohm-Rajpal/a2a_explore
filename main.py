from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    return {
        "name": "Math Helper Agent",
        "description": "An agent that can solve elementary math problems",
        "capabilities": ["chat", "math"],
        "version": "1.0"
    }

@app.post("/")
async def handle_a2a_message(request: Request):
    try:
        body = await request.body()
        print(f"📨 Raw request: {body.decode()}")
        
        data = await request.json()
        print(f"📋 Parsed data: {json.dumps(data, indent=2)}")
        
        # Extract user message from the correct A2A format
        user_text = ""
        if "params" in data and "message" in data["params"]:
            message = data["params"]["message"]
            if "parts" in message:
                for part in message["parts"]:
                    if part.get("kind") == "text" and "text" in part:
                        user_text += part["text"]
        
        print(f"🔤 Extracted text: '{user_text}'")
        
        if user_text:
            # Use OpenAI to solve the math problem
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a calculator. Solve the math problem and return ONLY the numerical answer. No words, no explanation, just the number."
                    },
                    {
                        "role": "user",
                        "content": user_text
                    }
                ],
                max_tokens=10,
                temperature=0
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"🧮 Answer: {answer}")
            
            # Return in A2A format
            result = {
                "jsonrpc": "2.0",
                "result": {
                    "messageId": "response-" + data.get("id", "1"),
                    "role": "assistant",
                    "parts": [{"kind": "text", "text": answer}]
                },
                "id": data.get("id", "1")
            }
            
            print(f"📤 Sending: {json.dumps(result, indent=2)}")
            return result
        
        # Fallback response
        return {
            "jsonrpc": "2.0",
            "result": {
                "messageId": "fallback",
                "role": "assistant", 
                "parts": [{"kind": "text", "text": "Ready for math!"}]
            },
            "id": data.get("id", "1")
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

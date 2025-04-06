import subprocess
from query_engine import generate_query_engine

import asyncio

async def main(): 
    query_engine = await generate_query_engine()
    response = query_engine.query("What are the main steps of RAG apps?")
    print(response)
    
try:
    result = subprocess.run(["curl", "http://localhost:11434"], capture_output=True, timeout=2)
    if result.returncode == 0:
        print("✅ Ollama server is already running!")
    else:
        print("❌ Ollama not reachable, starting it...")
        subprocess.Popen(["ollama", "serve"])
except Exception as e:
    print("Error checking Ollama:", e)
    subprocess.Popen(["ollama", "serve"])
    
asyncio.run(main=main())
    
    
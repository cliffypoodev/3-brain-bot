import os
import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# This allows your Base44 UI to talk to this script
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

async def call_brain(model, prompt, system_instruction):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=60.0
        )
        return response.json()['choices'][0]['message']['content']

@app.post("/chat")
async def chat_orchestrator(request: Request):
    data = await request.json()
    user_prompt = data.get("prompt")

    # BRAIN 1 & 2 run at the same time to save time!
    artist_task = call_brain(
        "mistralai/mistral-large", 
        user_prompt, 
        "You are an artist. Focus on creative prose, tone, and elegance."
    )
    scientist_task = call_brain(
        "deepseek/deepseek-chat", 
        user_prompt, 
        "You are a scientist. Focus on technical facts, logic, and data accuracy."
    )

    artist_response, scientist_response = await asyncio.gather(artist_task, scientist_task)

    # BRAIN 3 merges them together
    final_polish_prompt = f"Combine these two perspectives into one perfect, 'beefed up' response.\n\nCreative Draft: {artist_response}\n\nTechnical Facts: {scientist_response}"
    
    final_output = await call_brain(
        "qwen/qwen-2.5-72b-instruct", 
        final_polish_prompt, 
        "You are the Master Editor. Merge the creative and technical into a polished masterpiece."
    )

    return {"response": final_output}

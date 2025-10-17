from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Initialize the FastAPI lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events"""
    print("🚀 FastAPI starting up...")
    load_dotenv()
    global summarizer_chain
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
        prompt_template = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text in one concise line: {text}"
        )
        summarizer_chain = LLMChain(llm=llm, prompt=prompt_template)
        print("✅ Text summarizer model loaded successfully!")
    except Exception as e:
        print(f"❌ Error initializing summarizer model: {e}")
    yield
    print("🛑 FastAPI shutting down...")

# --- App Setup ---
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "user": "admin"})

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request, "user": "admin"})

@app.get("/default-instructions", response_class=HTMLResponse)
async def default_instructions_page(request: Request):
    return templates.TemplateResponse("default_instructions.html", {"request": request, "user": "admin"})

@app.get("/rules", response_class=HTMLResponse)
async def rules_page(request: Request):
    return templates.TemplateResponse("rules.html", {"request": request, "user": "admin"})

# --- Chat Endpoint ---
@app.post("/chat")
async def chat(user_query: str = Form(""), image: UploadFile = File(None)):
    if image and image.filename:
        raise HTTPException(status_code=400, detail="Image processing is not supported. Please provide text for summarization.")
    elif user_query:
        try:
            summary = await summarizer_chain.arun(text=user_query)
            return JSONResponse({"response": summary})
        except Exception as e:
            return JSONResponse({"response": f"Error summarizing text: {e}"})
    else:
        raise HTTPException(status_code=400, detail="No query or image provided.")

# --- Run App ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8002))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)

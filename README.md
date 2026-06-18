# DocuMind AI

> Chat with your documents using AI. Upload a PDF or text file, ask questions in plain English, and get answers with exact page citations — powered entirely by free tools.

---

## What this project does

Most businesses have knowledge locked inside documents — product manuals, legal contracts, HR policies, research reports. Reading through hundreds of pages to find one answer wastes time. DocuMind lets users **talk to their documents** instead.

You upload a file. The AI reads and indexes it. You ask a question like *"What is the refund policy?"* or *"Summarize section 3"* — and the AI answers instantly, telling you exactly which page it pulled the answer from.

This is called **RAG (Retrieval-Augmented Generation)** — the most in-demand AI pattern in freelancing right now.

---

## Live demo flow

```
User uploads contract.pdf
       ↓
DocuMind splits it into chunks and embeds them locally
       ↓
User asks: "When does this contract expire?"
       ↓
DocuMind searches the chunks, finds the relevant clause
       ↓
Groq LLM reads that clause and answers: "The contract expires on December 31, 2025 (Page 4)"
```

---

## Tech stack — 100% free

| Layer | Tool | Why free |
|---|---|---|
| LLM | Groq API (`llama3-8b-8192`) | Free tier — 14,400 tokens/min, no credit card |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Runs locally on your laptop, no API key |
| Vector database | ChromaDB | Saves to disk, no account, no server |
| Framework | LangChain | Open source |
| UI | Streamlit | Open source |

No OpenAI. No Pinecone. No paid services of any kind.

---

## Project structure

```
documind-ai/
│
├── app.py                  ← Run this. The full Streamlit UI.
├── config.py               ← All settings in one place.
│
├── core/
│   ├── loader.py           ← Reads PDF and TXT files from disk.
│   ├── splitter.py         ← Breaks documents into overlapping chunks.
│   ├── vectorstore.py      ← Embeds chunks and saves to ChromaDB.
│   ├── chain.py            ← The RAG logic: retrieve + generate answer.
│   └── pipeline.py         ← Wires loader → splitter → vectorstore together.
│
├── data/                   ← Created automatically at runtime.
│   ├── uploads/            ← Uploaded files saved here.
│   └── vectorstore/        ← ChromaDB database saved here.
│
├── .env                    ← Your secret keys. Never share or commit this.
├── .env.example            ← Template. Copy this to .env.
├── requirements.txt        ← All Python packages.
└── README.md
```

**Why this structure?**
Flat and simple. Each file has one job. You can explain every file to a client in one sentence. No nested `app/core/api/v1/` folders that confuse everyone including you.

---

## Setup — step by step

### Step 1: Clone the repo

```bash
git clone https://github.com/your-username/documind-ai.git
cd documind-ai
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3: Install packages

```bash
pip install -r requirements.txt
```

This installs everything. The HuggingFace embedding model (~90MB) downloads automatically the first time you run the app. After that it is cached locally.

### Step 4: Get your free Groq API key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Click **Sign Up** — no credit card required
3. Go to **API Keys** → **Create API Key**
4. Copy the key

### Step 5: Configure your environment

```bash
cp .env.example .env
```

Open `.env` and paste your key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

That is the only thing you need to fill in.

### Step 6: Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## How to use the app

**1. Upload a file**
In the right panel, click the upload area and select a PDF or TXT file. Multiple files are supported.

**2. Index the documents**
Click the **Index Documents** button. A progress bar shows each file being processed. This runs the full pipeline: load → split → embed → store. Takes 10–30 seconds depending on file size.

**3. Ask questions**
Type any question in the chat input at the bottom left. The AI searches your indexed documents and responds with an answer. Source chips appear below each answer showing which file and page was used. Click **View source excerpts** to see the exact text passage.

**4. Ask follow-up questions**
DocuMind remembers the last 5 exchanges so you can ask follow-ups naturally. "What about section 4?" or "Give me more detail on that" work correctly.

**5. Reset when done**
- **Clear Chat** — wipes conversation history, keeps documents indexed
- **Clear DB** — removes all indexed documents and resets everything

---

## How it works under the hood

### Ingestion (write path)

When you click Index Documents, the pipeline runs three steps:

**Step 1 — Load**
`core/loader.py` reads the file using LangChain's `PyPDFLoader` (for PDFs) or `TextLoader` (for TXT). Each page becomes a `Document` object with `page_content` and `metadata` (filename, page number).

**Step 2 — Split**
`core/splitter.py` uses `RecursiveCharacterTextSplitter` to break each page into smaller chunks of 800 characters with 150 characters of overlap. The overlap prevents answers from being cut off at chunk boundaries. A 10-page PDF typically becomes 40–80 chunks.

**Step 3 — Embed and store**
`core/vectorstore.py` converts each chunk into a vector (a list of 384 numbers) using the HuggingFace MiniLM model running locally on your CPU. These vectors are stored in ChromaDB, which persists them to `data/vectorstore/` on disk.

### Retrieval (read path)

When you ask a question:

1. Your question is converted into a vector using the same embedding model
2. ChromaDB finds the 4 chunks whose vectors are most mathematically similar to your question's vector
3. Those 4 chunks are formatted into a context string with source labels
4. The context + your question + conversation history are sent to the Groq LLM
5. The LLM reads only the retrieved context (not its training data) and generates an answer
6. Source citations are extracted from the retrieved chunks and shown as chips

The key constraint in the system prompt is: *"Answer ONLY from the provided context. If the answer is not there, say so."* This prevents the AI from hallucinating answers from its training data.

---

## Configuration

All settings live in `config.py`. Change once, affects everything.

| Setting | Default | What it does |
|---|---|---|
| `LLM_MODEL` | `llama3-8b-8192` | The Groq model used for generation |
| `LLM_TEMPERATURE` | `0.1` | Lower = more factual, higher = more creative |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local model for converting text to vectors |
| `CHUNK_SIZE` | `800` | Characters per chunk. Larger = more context per chunk |
| `CHUNK_OVERLAP` | `150` | Characters shared between adjacent chunks |
| `TOP_K` | `4` | How many chunks to retrieve per question |

**Tuning tips:**

- If answers are missing context, increase `CHUNK_SIZE` to `1000` or `TOP_K` to `6`
- If answers are slow, decrease `TOP_K` to `3`
- For better accuracy (slower), change `LLM_MODEL` to `llama3-70b-8192`
- For non-English documents, change `EMBEDDING_MODEL` to `paraphrase-multilingual-MiniLM-L12-v2`

---

## Free model options on Groq

All of these are free on Groq's free tier:

| Model | Speed | Quality | Best for |
|---|---|---|---|
| `llama3-8b-8192` | Very fast | Good | Default — most use cases |
| `llama3-70b-8192` | Slower | Better | Complex documents, legal, technical |
| `mixtral-8x7b-32768` | Fast | Good | Long documents (32k context window) |
| `gemma2-9b-it` | Fast | Good | Conversational tone |

Change `LLM_MODEL` in `config.py` to switch.

---

## Supported file types

| Extension | Loader used | Notes |
|---|---|---|
| `.pdf` | `PyPDFLoader` | Page numbers extracted automatically |
| `.txt` | `TextLoader` | Plain text, UTF-8 encoding |

To add `.docx` support, install `pip install docx2txt` and add this to `core/loader.py`:

```python
from langchain_community.document_loaders import Docx2txtLoader

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,   # add this line
}
```

And add `".docx"` to `SUPPORTED_EXTENSIONS` in `config.py`.

---

## Data and privacy

All document processing happens on your machine:

- Uploaded files are saved to `data/uploads/` on your local disk
- Embeddings are computed locally by the HuggingFace model — no data sent to any server
- ChromaDB stores everything in `data/vectorstore/` on your local disk
- Only the question + retrieved text chunks are sent to Groq's API for answer generation

This is a major selling point with clients who handle sensitive documents. Their files never leave their machine except for the small text excerpt sent to the LLM.

---

## Deployment

### Run locally (development)

```bash
streamlit run app.py
```

### Deploy to Railway (free tier)

Railway gives you a free hosted URL. Good for demos and client delivery.

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

In the Railway dashboard, go to **Variables** and add:
```
GROQ_API_KEY = your_key_here
```

Your app gets a public URL like `https://documind-ai-production.up.railway.app`.

### Deploy to Streamlit Cloud (free)

1. Push your repo to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Under **Advanced settings → Secrets**, add:
```toml
GROQ_API_KEY = "your_key_here"
```
5. Click Deploy

Free, permanent URL, no server management.

---

## Common errors and fixes

**`GROQ_API_KEY is not set`**
You did not create a `.env` file or forgot to add the key. Run `cp .env.example .env` and paste your key.

**`No documents indexed yet`**
You clicked the chat input before indexing. Upload a file and click Index Documents first.

**`ModuleNotFoundError: No module named 'langchain_groq'`**
Run `pip install -r requirements.txt` again. Make sure your virtual environment is activated.

**First run is slow**
Normal. The HuggingFace embedding model (~90MB) downloads on first run. Subsequent runs load from cache in under 3 seconds.

**Answers are wrong or vague**
Try increasing `TOP_K` from 4 to 6 in `config.py`, or reduce `CHUNK_SIZE` to 500 for more precise retrieval. For better accuracy overall, switch to `llama3-70b-8192`.

**ChromaDB error on restart**
Occasionally ChromaDB's local files get corrupted. Delete `data/vectorstore/` and re-index your documents.

---

## Freelancing notes

This project is built to be client-deliverable. Here is how to package and price it:

**Basic gig — $150 to $300**
Single file upload, Streamlit UI, deployed to Railway or Streamlit Cloud. Deliver as a GitHub repo with a setup guide.

**Standard gig — $300 to $600**
Multi-file support, conversation memory, source citations, custom branding (replace DocuMind with client's company name and colors).

**Premium gig — $600 to $1500**
Everything above plus a REST API (FastAPI wrapper around the pipeline and chain), so the client can embed the chatbot into their own website or mobile app.

**Upsell ideas:**
- Web scraping (index a website instead of a file) — add `WebBaseLoader`
- Streaming answers (words appear one by one) — replace `.invoke()` with `.stream()`
- Multi-language support — swap embedding model
- Scheduled re-indexing (re-index documents automatically every week)

---

## License

MIT — free to use, modify, and sell in client projects.
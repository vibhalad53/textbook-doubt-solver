# 📘 Textbook Doubt Solver

**Your AI-powered study companion.**

| | |
|---|---|
| **Problem Statement No.** | 33 |
| **Roll No.** | 33 |
| **Domain** | Generative AI / Large Language Models |
| **Architecture** | Retrieval-Augmented Generation (RAG) |
| **Tech stack** | Python · Streamlit · Google Gemini API · Sentence Transformers · FAISS · pypdf |

A student uploads a textbook chapter (PDF) and asks doubts about it. The app finds the
three most relevant passages in the chapter, asks Gemini to answer **using only those
passages**, and shows the answer together with a **quoted source line** and its
**page number**. An **Explain Simpler** button rewrites the last answer in easier words.

---

## 1. Architecture

```
 ┌──────────────┐   pypdf    ┌──────────────┐  split   ┌──────────────┐
 │  PDF upload  │ ─────────► │  Page texts  │ ───────► │    Chunks    │
 └──────────────┘            │ (page no. ✔) │          │ (page no. ✔) │
                             └──────────────┘          └──────┬───────┘
                                                               │ Sentence Transformers
                                                               ▼
 ┌──────────────┐  embed +   ┌──────────────┐          ┌──────────────┐
 │   Question   │ ─────────► │ FAISS search │ ◄─────── │ FAISS index  │
 └──────────────┘  search    │   (top 3)    │          └──────────────┘
                             └──────┬───────┘
                                    │ context + question + system prompt
                                    ▼
                             ┌──────────────┐  JSON: answer, quote, page
                             │    Gemini    │ ───────────────┐
                             └──────────────┘                ▼
                                                   ┌──────────────────────┐
                                                   │ Quote check: is the  │
                                                   │ quote really in the  │
                                                   │ PDF? Page taken from │
                                                   │ the chunk metadata.  │
                                                   └──────────┬───────────┘
                                                              ▼
                                            Answer · Source quote · Page no.
                                            [ Explain Simpler ] [ Clear Chat ]
```

**Two phases**

1. **Indexing (once per PDF):** extract text page by page → split into ~800-character
   chunks that never cross a page → turn each chunk into a 384-number embedding →
   store all embeddings in a FAISS index.
2. **Question answering (every question):** embed the question → FAISS returns the 3
   closest chunks → Gemini writes an answer from only those chunks → the app verifies the
   quotation word-for-word against the PDF and shows the page from the chunk's metadata.

**Why the quote check matters:** language models sometimes "improve" or invent quotes.
This app never shows a quote that is not in the PDF. If Gemini's quote can't be found
word-for-word, the app picks the closest real sentence from the retrieved text and
labels it *"Closest matching line, auto-selected from the textbook."* The page number
always comes from the PDF itself, never from the model.

---

## 2. Folder structure

```
textbook_doubt_solver/
├── app.py                 # Streamlit user interface (upload, chat, buttons)
├── rag_pipeline.py        # The 7 RAG functions + Gemini error handling
├── requirements.txt       # Python libraries to install
├── .env.example           # Template for your API key (copy to .env)
├── .gitignore             # Stops .env and junk files going to GitHub
├── README.md              # This file
├── sample_textbook.pdf    # 4-page sample chapter (Photosynthesis) for testing
├── .streamlit/
│   └── config.toml        # Blue/white/grey theme
├── screenshots/           # Put your own screenshots here
├── utils/
│   ├── __init__.py
│   ├── prompts.py         # System prompt, answer prompt, simplify prompt
│   └── styles.py          # Custom CSS for the dashboard look
└── docs/
    ├── PROJECT_REPORT.md   # Two-page procedure report
    ├── TEST_CASES.md       # Test cases with expected outputs
    ├── SCREENSHOTS.md      # Screenshots to take for submission
    └── VIVA_QUESTIONS.md   # Viva questions and answers
```

### The seven required functions (all in `rag_pipeline.py`)

| # | Function | What it does, in simple words |
|---|---|---|
| 1 | `extract_text_from_pdf()` | Reads every page with pypdf and keeps the page number with its text. Detects empty, locked, broken and scanned PDFs. |
| 2 | `split_text_into_chunks()` | Cuts each page into pieces of about 800 characters at sentence ends, with 150 characters of overlap. A chunk never spans two pages, so its page number is exact. |
| 3 | `create_embeddings()` | Uses the free `all-MiniLM-L6-v2` Sentence Transformer to turn text into vectors (lists of numbers that capture meaning). |
| 4 | `build_faiss_index()` | Puts the vectors into a FAISS `IndexFlatIP` index for fast similarity search. |
| 5 | `retrieve_relevant_chunks()` | Turns the question into a vector and asks FAISS for the 3 most similar chunks. |
| 6 | `generate_answer()` | Sends the 3 chunks + question + system prompt to Gemini, reads its JSON reply, and verifies the quote and page. |
| 7 | `explain_simpler()` | Asks Gemini to rewrite the previous answer in easier words without adding new information. |

---

## 3. Get a free Gemini API key

1. Open **https://aistudio.google.com/apikey** and sign in with a Google account.
2. Accept the terms if asked, then click **Create API key**.
3. Copy the key (a long string of letters and numbers).
4. Keep it private: don't paste it into code, screenshots or GitHub.

The free tier has per-minute and per-day request limits. If you see *"The free Gemini
limit has been reached"*, wait a minute and try again.

---

## 4. Run the project in VS Code (step by step)

**You need:** Python **3.10, 3.11 or 3.12** (from python.org; on Windows tick *"Add
Python to PATH"* during install), VS Code with the **Python** extension, and an internet
connection for the first run.

1. **Open the folder:** VS Code → *File → Open Folder…* → choose `textbook_doubt_solver`.
2. **Open a terminal:** *Terminal → New Terminal*.
3. **Create a virtual environment** (a private box for this project's libraries):

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
   You should now see `(venv)` at the start of the terminal line. If PowerShell blocks the
   activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

4. **Install the libraries** (takes a few minutes; PyTorch is large):

   ```bash
   pip install -r requirements.txt
   ```

5. **Add your API key:** copy `.env.example`, rename the copy to `.env`, and replace
   `your_gemini_api_key_here` with your key:

   ```
   GEMINI_API_KEY=AIza...your-real-key...
   GEMINI_MODEL=gemini-flash-latest
   ```

6. **Choose the interpreter:** press `Ctrl+Shift+P` → *Python: Select Interpreter* → pick
   the one inside `venv`.
7. **Run the app:**

   ```bash
   streamlit run app.py
   ```
   Your browser opens at **http://localhost:8501**. The first PDF takes longer because the
   embedding model (~90 MB) downloads once.
8. **Try it:** upload `sample_textbook.pdf`, wait for **● Ready to answer**, and ask
   *"What is photosynthesis?"*
9. **Stop the app:** press `Ctrl+C` in the terminal.

---

## 5. How to use the app

1. Upload a text-based PDF chapter in section **1 · Upload your textbook chapter**.
2. The status box shows each step: extracting → chunking → embeddings → FAISS index.
3. Type a doubt in the chat box at the bottom and press Enter.
4. Each answer shows:
   - **📝 Answer** (formatted explanation)
   - **📑 Source from textbook** (quoted line + page number)
   - **🔍 Retrieved textbook chunks** (expand to see the 3 chunks FAISS found)
5. Click **✨ Explain Simpler** to get an easier version of the last answer.
6. Click **🗑️ Clear Chat** to start over. Uploading a different PDF resets everything.

---

## 6. Upload to a public GitHub repository

1. Create a free account at **https://github.com** if you don't have one.
2. Install Git from **https://git-scm.com** and set your name once:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
3. On GitHub click **New repository** → name it `textbook-doubt-solver` → choose
   **Public** → don't add a README (you already have one) → **Create repository**.
4. In the VS Code terminal, inside the project folder:
   ```bash
   git init
   git add .
   git status            # CHECK: .env and venv/ must NOT appear in the list
   git commit -m "Textbook Doubt Solver - Problem 33, Roll 33"
   git branch -M main
   git remote add origin https://github.com/<your-username>/textbook-doubt-solver.git
   git push -u origin main
   ```
   When Git asks for a password, use a **Personal Access Token** (GitHub → Settings →
   Developer settings → Personal access tokens), not your account password. You can also
   sign in through the browser window that Git opens.
5. Refresh the GitHub page and check that `.env` is **not** there. If you ever push a
   key by mistake, delete it in Google AI Studio and create a new one.
6. Add your screenshots to `screenshots/`, then:
   ```bash
   git add screenshots
   git commit -m "Add screenshots"
   git push
   ```

---

## 7. Error handling built in

| Situation | What the student sees |
|---|---|
| No API key | Sidebar warning + password box; chat explains how to add the key |
| Wrong / expired key | "Your Gemini API key was rejected…" |
| Free quota used up | "The free Gemini limit has been reached for now…" |
| Model name retired | App automatically tries `gemini-flash-latest`, then `gemini-flash-lite-latest` |
| No internet | "Could not reach Gemini. Please check your internet connection." |
| Scanned (image-only) PDF | "No readable text was found… probably a scanned document…" |
| Empty / broken / locked PDF | A clear message for each case |
| Question not in the chapter | "This information is not available in the uploaded textbook." |

---

## 8. Troubleshooting

- **`streamlit` is not recognized** → the virtual environment isn't active. Run the
  activate command from step 3 again.
- **`pip install` fails on faiss-cpu or torch** → use Python 3.10–3.12, then run
  `python -m pip install --upgrade pip` and try again.
- **Model not found error** → open `.env` and set `GEMINI_MODEL` to a model listed at
  https://ai.google.dev/gemini-api/docs/models
- **Page numbers look off** → the app shows the page number as counted in a PDF viewer.
  If the PDF stores printed page numbers (e.g. "page 45" of a book), the app shows those too.

---

## 9. Limitations

- Scanned PDFs need OCR first; the app doesn't do OCR.
- Tables and equations in PDFs sometimes extract as jumbled text.
- Answers are only as good as the retrieved chunks; very broad questions may miss parts
  of the chapter.
- The FAISS index lives in memory, so each new session re-processes the PDF.

## 10. Future scope

OCR for scanned books, multiple chapters at once, chat memory for follow-up questions,
saving indexes to disk, and quiz generation from the chapter.

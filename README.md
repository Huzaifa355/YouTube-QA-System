

# 🎯 YouTube Video Question Answering System (RAG-based)

This project is an interactive Streamlit application that answers user questions based on the **transcript of a YouTube video** using a **Retrieval-Augmented Generation (RAG)** pipeline.

Users provide a YouTube link, select a transcript language, and input a question. The system extracts the transcript, processes it, retrieves relevant chunks using vector similarity, and then generates context-aware answers using a large language model (LLM).


## 🚀 Features

* ✅ Accepts any YouTube video link
* 🌐 Supports multiple transcript languages (with suggestions)
* 🔍 RAG pipeline with FAISS and transformer-based embeddings
* 🧠 LLM integration (LLaMA 3 via ChatTogether) for answer generation
* ⚙️ Customizable retrieval size (`k`)
* 📄 Expandable view of retrieved transcript chunks
* 🛡️ Graceful error handling for missing captions or service issues


## 🧰 Tech Stack

* **Frontend:** Streamlit
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Store:** FAISS
* **LLM:** Meta LLaMA 3 (via LangChain + ChatTogether)
* **Transcript Extraction:** `youtube-transcript-api`
* **Prompting & Chaining:** LangChain


## 📦 Installation

### 1. Clone the repository

git clone https://github.com/Huzaifa355/YouTube-QA-System.git


### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Example requirements:
>
> ```txt
> streamlit
> langchain
> langchain-community
> langchain-together
> langchain-huggingface
> faiss-cpu
> youtube-transcript-api
> python-dotenv
> ```

### 3. Add your `.env` file

Create a `.env` file and include your API key(s) if needed:

```env
TOGETHER_API_KEY=your_key_here
```



## ▶️ Run the App

```bash
streamlit run streamlit-app.py
```


## 🧠 How it Works

1. **Transcript Extraction**
   Uses `youtube-transcript-api` to fetch video captions in the selected language.

2. **Chunking & Embedding**
   Splits transcript into overlapping chunks → embeds with `all-MiniLM-L6-v2`.

3. **Vector Search (FAISS)**
   Retrieves top-k most similar chunks to the user question.

4. **LLM Query**
   Uses a custom prompt template to generate a grounded answer using LLaMA 3.


## 📌 To Do

* [ ] Add multi-video memory/chat history
* [ ] Support file/document input as alternative
* [ ] Add streaming response support
* [ ] Optional OpenAI model fallback


## 👨‍💻 Author

**Huzaifa Shafique**

* 🎓 National Textile University
* 📧 [huzaifashafique355@gmail.com](mailto:huzaifashafique355@gmail.com)
* 🌐 [GitHub](https://github.com/Huzaifa355) | [LinkedIn](https://www.linkedin.com/in/huzaifa-shafique355)



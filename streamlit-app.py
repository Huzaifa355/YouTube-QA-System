import os
import re
import difflib
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_together import ChatTogether
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Video QA (RAG)", layout="wide")
st.title("📺 Video Question Answering (RAG)")

# Sidebar inputs
st.sidebar.header("Settings")
video_url = st.sidebar.text_input("YouTube Video URL", "https://www.youtube.com/watch?v=Gfr50f6ZBvo")

# Language selection with suggestions
language_map = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Arabic": "ar",
    "Chinese": "zh"
}
lang_option = st.sidebar.selectbox("Transcript Language", options=list(language_map.keys()) + ["Other"])
if lang_option == "Other":
    custom_lang = st.sidebar.text_input("Enter language code (e.g., 'en')", "en")
    # Show suggestions based on common codes
    codes = list(language_map.values())
    suggestions = difflib.get_close_matches(custom_lang, codes, n=5, cutoff=0.1)
    if suggestions:
        st.sidebar.write("Did you mean:", ", ".join(suggestions))
    transcript_lang = custom_lang
else:
    transcript_lang = language_map[lang_option]

k = st.sidebar.number_input("Number of similar chunks (k)", min_value=1, max_value=10, value=4, step=1)
question = st.sidebar.text_area("Your Question", "What is deepmind?")
run_button = st.sidebar.button("Run QA")

# Utility to extract video id
def extract_video_id(url: str) -> str:
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else url

if run_button:
    if not video_url or not question:
        st.error("Please provide both a video URL and a question.")
    else:
        with st.spinner("Processing transcript and building RAG chain..."):
            video_id = extract_video_id(video_url)
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[transcript_lang])
                transcript = " ".join(chunk['text'] for chunk in transcript_list)
            except TranscriptsDisabled:
                st.error("No captions available for this video in the selected language.")
                st.stop()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.create_documents([transcript])

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(docs, embeddings)

            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": int(k)})
            llm = ChatTogether(model="meta-llama/Llama-3-70b-chat-hf", temperature=0.2)

            prompt = PromptTemplate(
                template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}
""",
                input_variables=['context', 'question']
            )

            retrieved_docs = retriever.invoke(question)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            final_input = prompt.invoke({"context": context_text, "question": question})

            try:
                answer = llm.invoke(final_input)
            except Exception as e:
                st.error(f"LLM service error: {e}")
                st.stop()

        # Display outputs
        st.subheader("🎯 Answer")
        st.write(answer.content)

        with st.expander("🔍 Retrieved Chunks"):
            for i, doc in enumerate(retrieved_docs, 1):
                snippet = doc.page_content[:200].strip().replace("\n", " ")
                st.markdown(f"**Chunk {i}:** {snippet}...")

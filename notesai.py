import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import os
from openai import OpenAI
import yt_dlp
import imageio_ffmpeg
from groq import Groq
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from bs4 import SoupStrainer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import tempfile

st.set_page_config(
    page_title="NotesAI - Smart Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .source-card {
        background: #f8f9fa;
        color:  #212529;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #d1ecf1;   
        border-left: 4px solid #0c5460;
        color: #0c5460;
    }
    .bot-message {
        background: #fff3cd;   /* soft yellow */
        border-left: 4px solid #856404;
        color: #856404; 
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .sidebar-info {
        background: #f0f2f6;
        color: #212529;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'retrieval_chain' not in st.session_state:
    st.session_state.retrieval_chain = None
if 'processed_sources' not in st.session_state:
    st.session_state.processed_sources = []

# API Keys setup
def setup_api_keys():
    try:
        # Get API keys from Streamlit secrets
        openai_key = st.secrets["OPENAI_API_KEY"]
        groq_key = st.secrets["GROQ_API_KEY"]
        
        # Store in session state
        st.session_state.openai_key = openai_key
        st.session_state.groq_key = groq_key
        
        return True
    except KeyError as e:
        st.error(f"❌ Missing API key in secrets: {e}")
        st.info("Please configure your API keys in the secrets.toml file.")
        return False
    except Exception as e:
        st.error(f"❌ Error loading API keys: {e}")
        return False

# Your existing functions with minor fixes
@st.cache_data(show_spinner=False)
def fetch_transcript_YT(video_url: str) -> Document:
    if 'groq_key' not in st.session_state:
        st.error("Groq API key not configured!")
        return None
        
    client = Groq(api_key=st.session_state.groq_key)
    parsed_url = urlparse(video_url)

    if "youtube" in parsed_url.netloc:
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            video_id = query_params["v"][0]
        else:
            video_id = parsed_url.path.split("/")[-1]
    elif "youtu.be" in parsed_url.netloc:
        video_id = parsed_url.path.lstrip("/")
    else:
        st.error("Invalid YouTube URL")
        return None

    # Method 1: YouTube Transcript API
    try:
        ytt_api = YouTubeTranscriptApi()
        result = ytt_api.fetch(video_id)
        text = " ".join(snippet.text for snippet in result.snippets)
        if text.strip():
            return Document(page_content=text, metadata={"source": video_url})
    except Exception as e:
        st.warning(f"YouTubeTranscriptApi failed: {e}")

    # Method 2: Groq Whisper
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = os.path.join(temp_dir, "audio.mp3")
            
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": audio_path.replace('.mp3', '.%(ext)s'),
                "quiet": True,
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }],
                "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe()
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            # Find the actual audio file
            for file in os.listdir(temp_dir):
                if file.endswith('.mp3'):
                    audio_path = os.path.join(temp_dir, file)
                    break

            if os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-large-v3-turbo",
                        file=f
                    )
                text = transcription.text
                return Document(page_content=text, metadata={"source": video_url})
            else:
                st.error("Audio file not found after download")
                return None

    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None

@st.cache_data(show_spinner=False)
def fetch_web(url: str) -> list[Document]:
    try:
        only_content = SoupStrainer("p")
        loader = WebBaseLoader(url, bs_kwargs={"parse_only": only_content})
        docs = loader.load()
        if not docs:
            st.warning("No content could be extracted from this webpage due to some restricions from web Page side. Try another one")
            return []
        return docs
    except Exception as e:
        st.error(f"Failed to fetch web content: {e}")
        return []

@st.cache_data(show_spinner=False)
def fetch_pdf(uploaded_file) -> Document:
    try:
        pdf_reader = PdfReader(uploaded_file)
        content = ""
        for page in pdf_reader.pages:
            content += page.extract_text() or ""
        text = content.strip()
        return Document(page_content=text, metadata={'source': uploaded_file.name})
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

@st.cache_resource(show_spinner=False)
def process_docs(docs: list[Document]) -> FAISS:
    if 'openai_key' not in st.session_state:
        st.error("OpenAI API key not configured!")
        return None
        
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        document_chunks = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_key)
        db = FAISS.from_documents(document_chunks, embeddings)
        return db
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        return None

def setup_bot(docs: list[Document]):
    if 'groq_key' not in st.session_state:
        st.error("Groq API key not configured!")
        return None, None
        
    db = process_docs(docs)
    if not db:
        return None, None
    if not docs:
        st.error(" Some issue with this webpage. Please try another one.")
        return None, None
        
    memory = ConversationBufferMemory(return_messages=True)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=st.session_state.groq_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert tutor helping students learn. "
         "Your role is to explain concepts step by step, give examples, "
         "and make complex ideas simple. "
         "Always base your answers ONLY on the provided context. "
         "If the context does not contain the answer, say you don't know "
         "instead of making up information. "
         "If a student asks a question about the context, answer it clearly and directly using the context. "
         "If a student asks you to teach or explain from a source, then teach them thoroughly using the context "
         "with examples, summaries, and structured explanations. "
         "Context: {context}"
        ),
        MessagesPlaceholder(variable_name="memory"),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain,memory

# Main UI
def main():
    # Header
    st.markdown('<h1 class="main-header">📚 NotesAI - Smart Learning Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Check API keys
    if not setup_api_keys():
        return

    # Main content area with tabs
    tab1, tab2 = st.tabs(["📁 Add Sources", "💬 Chat with AI Tutor"])
    
    with tab1:
        st.header("📁 Add Learning Sources")
        
        # Source type selection
        col1, col2 = st.columns([1, 2])
        with col1:
            source_type = st.selectbox(
                "Choose source type:",
                ["YouTube Video", "Web Page", "PDF Document"],
                key="source_type_select"
            )
        
        all_docs = []
        
        if source_type == "YouTube Video":
            st.subheader("🎥 YouTube Video")
            youtube_url = st.text_input(
                "Enter YouTube URL:",
                placeholder="https://www.youtube.com/watch?v=...",
                key="youtube_input"
            )
            
            if st.button("Process YouTube Video", key="process_youtube") and youtube_url:
                with st.spinner("Fetching transcript..."):
                    doc = fetch_transcript_YT(youtube_url)
                    if doc:
                        all_docs.append(doc)
                        st.success("✅ YouTube video processed!")
                        st.session_state.processed_sources.append(f"YouTube: {youtube_url}")

        elif source_type == "Web Page":
            st.subheader("🌐 Web Page")
            web_url = st.text_input(
                "Enter webpage URL:",
                placeholder="https://example.com/article",
                key="web_input"
            )
            
            if st.button("Process Web Page", key="process_web") and web_url:
                with st.spinner("Fetching web content..."):
                    docs = fetch_web(web_url)
                    if docs:
                        all_docs.extend(docs)
                        st.success("✅ Web page processed!")
                        st.session_state.processed_sources.append(f"Web: {web_url}")

        elif source_type == "PDF Document":
            st.subheader("📄 PDF Document")
            uploaded_file = st.file_uploader(
                "Upload PDF file:",
                type="pdf",
                key="pdf_uploader"
            )
            
            if st.button("Process PDF", key="process_pdf") and uploaded_file:
                with st.spinner("Processing PDF..."):
                    doc = fetch_pdf(uploaded_file)
                    if doc:
                        all_docs.append(doc)
                        st.success("✅ PDF processed!")
                        st.session_state.processed_sources.append(f"PDF: {uploaded_file.name}")

        # Setup bot if documents are available
        if all_docs:
            with st.spinner("Setting up AI tutor..."):
                chain, memory = setup_bot(all_docs)
                if chain and memory:
                  st.session_state.retrieval_chain = chain
                  st.session_state.memory = memory
                  st.success("🤖 AI Tutor is ready! Switch to the Chat tab to start asking questions.")
                else:
                  st.error("⚠️ Could not set up AI tutor because this page isn't allowing extraction of content. Please try another webpage URl.")  

        # Display processed sources
        if st.session_state.processed_sources:
            st.markdown("### 📋 Processed Sources")
            for i, source in enumerate(st.session_state.processed_sources):
                st.markdown(f"<div class='source-card'>✅ {source}</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear All Sources", key="clear_sources"):
                    st.session_state.processed_sources = []
                    st.session_state.retrieval_chain = None
                    st.session_state.chat_history = []
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()
            
            with col2:
                if st.session_state.retrieval_chain:
                    st.info("✅ AI Tutor ready! Go to Chat tab →")

        # Welcome message if no sources
        if not st.session_state.processed_sources:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
                <h3>🚀 Getting Started:</h3>
                <ol style="text-align: left; display: inline-block; color: #2c3e50;">
                    <li>Choose a source type above</li>
                    <li>Add your learning materials</li>
                    <li>Process the content</li>
                    <li>Switch to "Chat with AI Tutor" tab after document processing to start learning!</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

            # Feature highlights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="source-card">
                    <h4 style="color: #2c3e50;">🎥 YouTube Videos</h4>
                    <p style="color: #34495e;">Extract transcripts and learn from educational videos</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="source-card">
                    <h4 style="color: #2c3e50;">🌐 Web Articles</h4>
                    <p style="color: #34495e;">Process web pages and articles for learning</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="source-card">
                    <h4 style="color: #2c3e50;">📄 PDF Documents</h4>
                    <p style="color: #34495e;">Upload and learn from PDF documents</p>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        # Sidebar info about processed sources
        with st.sidebar:
            st.header("📊 Session Info")
            if st.session_state.processed_sources:
                st.markdown("### 📋 Active Sources")
                for source in st.session_state.processed_sources:
                    st.markdown(f"<div class='sidebar-info'>✅ {source}</div>", unsafe_allow_html=True)
                
                if st.session_state.retrieval_chain:
                    st.success("🤖 AI Tutor: Ready")
                else:
                    st.warning("⚠️ AI Tutor: Setting up...")
            else:
    # No sources available message
                st.markdown("""
                <div style="text-align: left; padding: 1.5rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
                <h3 style="color: #2c3e50;">📌 Instructions for Adding Sources</h3>
                <ul style="color: #34495e; font-size: 1rem; line-height: 1.6;">
                     <li><b>📄 PDF:</b> Use only unencrypted and unlocked PDFs. Avoid scanned or image-only PDFs.</li>
                    <li><b>🌐 Web Page:</b> Some sites may block text extraction. If content isn’t loading, try another webpage.</li>
                    <li><b>🎥 YouTube Video:</b> If subtitles are missing, long videos may take more time to process.</li>
                </ul>
                <p style="color: #2c3e50; margin-top: 1rem;">
                         👉 Go to the <b>'Add Sources'</b> tab to upload your content and start learning!
                 </p>
                </div>
                """, unsafe_allow_html=True)
# Chat interface
        if st.session_state.retrieval_chain:
            st.header("💬 Chat with your AI Tutor")
            
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message bot-message"><strong>AI Tutor:</strong> {message["content"]}</div>', 
                              unsafe_allow_html=True)

            # Chat input
            user_question = st.text_input(
                "Ask a question about your sources:",
                placeholder="What are the main concepts explained in the content?",
                key="user_chat_input"
            )

            col1, col2 = st.columns([3, 1])
            with col1:
                send_button = st.button("Send Question", key="send_question")
            with col2:
                if st.button("🗑️ Clear Chat", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.rerun()

            if send_button and user_question:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Get AI response
                with st.spinner("AI Tutor is thinking..."):
                    try:
                        response = st.session_state.retrieval_chain.invoke({
                            "input": user_question,
                            "memory": st.session_state.memory.chat_memory.messages
                            })

                        bot_response = response.get("answer", "I couldn't generate a response.")
                        
                        # Add bot response to history
                        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                        st.session_state.memory.chat_memory.add_user_message(user_question)
                        st.session_state.memory.chat_memory.add_ai_message(bot_response)
                        
                        # Refresh the page to show new messages
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

            # Quick action buttons
            # Quick action buttons
            st.markdown("### 🚀 Quick Actions")
            col1, col2, col3 = st.columns(3)

            def ask_and_store(question: str):
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.spinner("AI Tutor is thinking..."):
                    try:
                        response = st.session_state.retrieval_chain.invoke({
                            "input": question,
                            "memory": st.session_state.memory.chat_memory.messages
                        })
                        bot_response = response.get("answer", "I couldn't generate a response.")

            # Add assistant response
                        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

            # ✅ Update memory
                        st.session_state.memory.chat_memory.add_user_message(question)
                        st.session_state.memory.chat_memory.add_ai_message(bot_response)

                        st.rerun()
                    except Exception as e:
                         st.error(f"Error generating response: {e}")

            with col1:
                if st.button("📝 Summarize Content", key="quick_summarize"):
                    ask_and_store("Please provide a comprehensive summary of all the content.")

            with col2:
                if st.button("🔍 Key Concepts", key="quick_concepts"):
                    ask_and_store("What are the main key concepts and ideas explained in the content?")

            with col3:
                if st.button("❓ Generate Quiz", key="quick_quiz"):
                    ask_and_store("Create a quiz with questions based on the content to test my understanding.")


        else:
            # No sources available message
            st.info("👈 Please add some learning sources first in the 'Add Sources' tab to start chatting with your AI tutor.")
            
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
                <h3 style="color: #2c3e50;">💡 How it works:</h3>
                <p style="color: #34495e;">
                    1. Add YouTube videos, web pages, or PDF documents<br>
                    2. Our AI processes and understands the content<br>
                    3. Ask questions and get detailed explanations<br>
                    4. Generate summaries, quizzes, and key concepts
                </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
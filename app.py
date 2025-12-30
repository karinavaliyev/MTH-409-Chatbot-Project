import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv

# Paketler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# API Anahtarlarını Yükle
load_dotenv()

# --- YAPILANDIRMA ---
CHROMA_DIR = "./.chroma_stardew"
COLLECTION = "stardew-knowledge" 
OPENAI_MODEL = "gpt-4o"
GEMINI_MODEL = "gemini-3-flash-preview"

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Stardew Valley Rehberi", 
    page_icon="🧑‍🌾", 
    initial_sidebar_state="collapsed"
)
st.title("🧑‍🌾 Stardew Valley Chatbot")

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# --- FONKSİYONLAR ---
def create_new_chat():
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_chat = {"id": chat_id, "title": "Yeni Sohbet", "messages": []}
    st.session_state.chat_history.insert(0, new_chat)
    st.session_state.current_chat_id = chat_id

def get_current_chat():
    if not st.session_state.current_chat_id:
        create_new_chat()
    for chat in st.session_state.chat_history:
        if chat["id"] == st.session_state.current_chat_id:
            return chat
    create_new_chat()
    return st.session_state.chat_history[0]

def delete_chat(chat_id):
    st.session_state.chat_history = [c for c in st.session_state.chat_history if c["id"] != chat_id]
    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    model_choice = st.radio("🤖 Model", ["OpenAI (GPT-4o)", "Google (Gemini)"], horizontal=True)
    
    st.divider()
    st.header("💬 Sohbet Geçmişi")
    
    if st.button("➕ Yeni Sohbet", use_container_width=True):
        create_new_chat()
        st.rerun()
    
    # Sohbet listesi
    for chat in st.session_state.chat_history:
        is_active = chat["id"] == st.session_state.current_chat_id
        col1, col2 = st.columns([6, 1])
        
        with col1:
            label = f"🔹 {chat['title']}" if is_active else f"💭 {chat['title']}"
            if st.button(label, key=f"chat_{chat['id']}", use_container_width=True):
                st.session_state.current_chat_id = chat["id"]
                st.rerun()
        
        with col2:
            if st.button("×", key=f"del_{chat['id']}"):
                delete_chat(chat["id"])
                st.rerun()

# --- VEKTÖR DB ---
@st.cache_resource
def get_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings, collection_name=COLLECTION)

vectorstore = get_vectorstore()

# --- MODEL ---
def get_llm(choice):
    if choice == "OpenAI (GPT-4o)":
        return ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)

# --- CHAT ---
current_chat = get_current_chat()

# Mesajları göster
for msg in current_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Kaynakları expander ile göster
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Kaynaklar"):
                st.markdown(msg["sources"])

# Sık sorulan sorular (sadece boş sohbette)
if not current_chat["messages"]:
    st.markdown("**💡 Sık Sorulan Sorular:**")
    c1, c2, c3 = st.columns(3)
    if c1.button("🎁 En iyi hediyeler?", use_container_width=True):
        st.session_state.pending_query = "Herkesin sevdiği evrensel hediyeler nelerdir?"
        st.rerun()
    if c2.button("🌾 Karlı ekinler?", use_container_width=True):
        st.session_state.pending_query = "En karlı ekinler hangileri?"
        st.rerun()
    if c3.button("💍 Evlenme rehberi?", use_container_width=True):
        st.session_state.pending_query = "Nasıl evlenebilirim?"
        st.rerun()

# Sorguyu al
query = st.session_state.pop("pending_query", None) or st.chat_input("Ruhlar bugün fısıldaşmayı seviyor…")

if query:
    # Başlığı güncelle
    if current_chat["title"] == "Yeni Sohbet":
        current_chat["title"] = query[:30] + ("..." if len(query) > 30 else "")
    
    # Kullanıcı mesajını ekle ve göster
    current_chat["messages"].append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Yıldızçiyi Vadisi rehberine bakılıyor..."):
            # Retrieval
            docs = vectorstore.similarity_search(query, k=6)
            context = "\n\n".join([d.page_content for d in docs])
            
            # Prompt
            system = (
                "Sen Stardew Valley oyununda uzman, samimi bir asistansın. Adın 'Stardew Rehberi'. "
                "Cevaplarını Türkçe ver ve emojiler kullan.\n\n"
                "KURALLAR:\n"
                "1. Selamlaşma (merhaba, selam, nasılsın vb.) → Sıcak karşılık ver ve yardımcı olmayı teklif et.\n"
                "2. Stardew Valley soruları → Aşağıdaki 'REHBER BİLGİLERİ'ni kullanarak cevapla.\n"
                "3. Oyunla ALAKASIZ sorular (politika, matematik, kodlama vb.) → "
                "'Ben sadece Stardew Valley hakkında yardımcı olabilirim! 🧑‍🌾' de ve oyunla ilgili soru sormaya teşvik et.\n"
                "4. Rehberde bilgi yoksa dürüstçe 'Bu konuda bilgim yok' de.\n\n"
                f"REHBER BİLGİLERİ:\n{context}"
            )
            
            # Mesaj listesi (son 10 mesaj)
            messages = [SystemMessage(content=system)]
            for m in current_chat["messages"][-10:]:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                else:
                    messages.append(AIMessage(content=m["content"]))
            
            # LLM çağır
            llm = get_llm(model_choice)
            response = llm.invoke(messages)
            
            # Cevabı al
            content = response.content
            if isinstance(content, list):
                answer = content[0].get('text', str(content))
            else:
                answer = content
            
            # Kaynakları formatla
            sources_text = []
            for i, doc in enumerate(docs):
                src = doc.metadata.get("source", "").lower()
                name = doc.metadata.get("name", "")
                if "pdf" in src:
                    page = doc.metadata.get("page", 0) + 1
                    sources_text.append(f"{i+1}. 📖 Stardew Valley Guide - Sayfa {page}")
                elif "character" in src:
                    sources_text.append(f"{i+1}. 👤 {name}")
                elif "crop" in src:
                    sources_text.append(f"{i+1}. 🌾 {name}")
                else:
                    sources_text.append(f"{i+1}. 📄 {name or 'Bilinmiyor'}")
            
            sources_formatted = "\n".join(sources_text)
            
            st.markdown(answer)
            
            # Kaynakları expander ile göster
            with st.expander("📚 Kaynaklar"):
                st.markdown(sources_formatted)
    
    # Asistan cevabını kaydet (kaynaklar ayrı)
    current_chat["messages"].append({"role": "assistant", "content": answer, "sources": sources_formatted})
    st.rerun()

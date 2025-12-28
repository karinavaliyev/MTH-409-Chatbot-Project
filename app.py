import streamlit as st
import os
from dotenv import load_dotenv

# Sadece hata vermeyen, çalışan paketleri kullanıyoruz
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# API Anahtarlarını Yükle
load_dotenv()

# --- YAPILANDIRMA ---
CHROMA_DIR = "./.chroma_stardew"
COLLECTION = "stardew-characters"
OPENAI_MODEL = "gpt-4o"

st.set_page_config(page_title="Stardew Valley Rehberi", page_icon="👨‍🌾")
st.title("👨‍🌾 Stardew Valley Uzmanı")

# --- 1. VEKTÖR VERİTABANINA BAĞLANMA ---
@st.cache_resource
def get_vectorstore():
    # OpenAIEmbeddings langchain-openai paketinden gelir, hata vermez
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION
    )

vectorstore = get_vectorstore()

# --- 2. MODELİ HAZIRLA ---
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)

# --- 3. STREAMLIT CHAT MANTIĞI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Pelikan Kasabası hakkında bir şeyler sor...")

if query:
    # Kullanıcı mesajını göster
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Rehber tozlu raflardan indiriliyor..."):
            
            # --- MANUEL RETRIEVAL (Arama) ---
            # Zincir kullanmadan doğrudan veritabanında arıyoruz
            docs = vectorstore.similarity_search(query, k=3)
            
            # Bulunan dökümanları tek bir metin haline getiriyoruz (Context)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # --- MANUEL GENERATION (Üretim) ---
            # Sistemi ve Soruyu modele gönderiyoruz
            system_instruction = (
                "Sen neşeli bir Stardew Valley rehberisin. "
                "Sorulan sorulara türkçe cevap ver."
                "Sadece aşağıdaki bilgileri kullanarak cevap ver. "
                "Bilgi yoksa 'Bilmiyorum' de. Emojiler kullan.\n\n"
                f"REHBER BİLGİLERİ:\n{context}"
            )
            
            messages = [
                SystemMessage(content=system_instruction),
                HumanMessage(content=query)
            ]
            
            # Modeli çağır
            response = llm.invoke(messages)
            answer = response.content
            
            st.markdown(answer)
            
            # Hangi karakterin bilgisini kullandığını görmek için:
            with st.expander("Göz atılan kaynaklar"):
                for doc in docs:
                    st.write(f"- {doc.metadata.get('character', 'Genel Bilgi')}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
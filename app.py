import streamlit as st
import os
from dotenv import load_dotenv

# Paketler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# API Anahtarlarını Yükle
load_dotenv()

# --- YAPILANDIRMA ---
CHROMA_DIR = "./.chroma_stardew"
# ÖNEMLİ DEĞİŞİKLİK: ingest.py'de kullandığımız yeni koleksiyon adını buraya yazıyoruz
COLLECTION = "stardew-knowledge" 
OPENAI_MODEL = "gpt-4o"

st.set_page_config(page_title="Stardew Valley Rehberi", page_icon="👨‍🌾")
st.title("👨‍🌾 Stardew Valley Chatbot")

# --- 1. VEKTÖR VERİTABANINA BAĞLANMA ---
@st.cache_resource
def get_vectorstore():
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

# Eski mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Pelikan Kasabası hakkında sor...")

if query:
    # Kullanıcı mesajını göster
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Çiftlik kayıtlarına bakılıyor..."):
            
            # --- MANUEL RETRIEVAL (Arama) ---
            # Veri havuzu büyüdüğü için 'k' değerini 3'ten 4'e çıkarabiliriz, daha fazla bağlam alsın
            docs = vectorstore.similarity_search(query, k=4)
            
            # Bulunan dökümanları birleştir
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # --- MANUEL GENERATION (Üretim) ---
            system_instruction = (
                "Sen neşeli bir Stardew Valley rehberisin. "
                "Sorulan sorulara yalnızca Türkçe cevap ver. "
                "Sadece aşağıdaki bilgileri kullanarak cevap ver. "
                "Eğer sorulan şey bağlamda yoksa uydurma, 'Bilmiyorum' de. "
                "Cevaplarında emoji kullan.\n\n"
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
            
            # Kaynakları göster (Metadata 'name' olarak güncellendiği için burayı düzelttim)
            with st.expander("Göz atılan kaynaklar"):
                for doc in docs:
                    # ingest.py'de metadata olarak "name" ve "source" kaydetmiştik
                    source_name = doc.metadata.get('name', 'Bilinmiyor')
                    source_type = doc.metadata.get('source', 'Genel')
                    st.write(f"- {source_type.upper()}: {source_name}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
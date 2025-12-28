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
COLLECTION = "stardew-knowledge" 
OPENAI_MODEL = "gpt-4o"

st.set_page_config(page_title="Stardew Valley Rehberi", page_icon="📘")
st.title("📘 Stardew Valley: Kapsamlı Rehber")

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

query = st.chat_input("Rehber kitaba ve kayıtlara sor...")

if query:
    # Kullanıcı mesajını göster
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Rehber kitap sayfaları ve çiftlik kayıtları taranıyor..."):
            
            # --- MANUEL RETRIEVAL (Arama) ---
            # PDF eklediğimiz için k=6 yapıyoruz. PDF parçaları bazen uzundur, 
            # net cevabı kaçırmamak için daha fazla bağlam çekmek iyidir.
            docs = vectorstore.similarity_search(query, k=6)
            
            # Bulunan dökümanları birleştir
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # --- MANUEL GENERATION (Üretim) ---
            system_instruction = (
                "Sen uzman bir Stardew Valley asistanısın. "
                "Eline hem karakter/ekin tabloları hem de detaylı bir rehber kitap geçti. "
                "Sorulan sorulara Türkçe cevap ver. "
                "Sadece aşağıdaki 'REHBER BİLGİLERİ' kısmındaki metinleri kullan. "
                "Eğer bilgi CSV tablolarından geliyorsa net rakamlar ver. "
                "Eğer bilgi Rehber Kitaptan (PDF) geliyorsa detaylı taktikler ver. "
                "Bağlamda bilgi yoksa dürüstçe 'Bilmiyorum' de. "
                "Cevaplarında uygun emojiler kullan (📖, 🥕, 🐟 vb).\n\n"
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
            
            # Kaynakları göster (Artık PDF kaynağını da göreceğiz)
            with st.expander("Göz atılan kaynaklar"):
                for doc in docs:
                    # Metadata kontrolü
                    source_type = doc.metadata.get('source', 'Genel').replace("csv_", "").replace("pdf_", "").upper()
                    name = doc.metadata.get('name', 'Bilinmiyor')
                    
                    # Kaynağa göre ikon ekleyelim
                    icon = "📄" if "GUIDE" in source_type else "📊"
                    
                    st.write(f"{icon} **{source_type}:** {name}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
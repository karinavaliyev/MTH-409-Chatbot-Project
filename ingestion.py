import pandas as pd
import os
import ast
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Ayarlar
DATA_FILE = "data/characters.csv"  # Yüklediğin dosya adı
CHROMA_DIR = "./.chroma_stardew"
COLLECTION = "stardew-characters"
EMBED_MODEL = "text-embedding-3-large"

def clean_list_str(s):
    """CSV içindeki ['Item1', 'Item2'] formatındaki stringleri temizler."""
    try:
        lst = ast.literal_eval(s)
        return ", ".join(lst) if lst else "None"
    except:
        return s

def process_stardew_csv(file_path):
    docs = []
    df = pd.read_csv(file_path)
    
    for _, row in df.iterrows():
        name = row['Name']
        # Ham veriyi anlamlı bir paragrafa dönüştürüyoruz
        content = (
            f"{name} is a resident of Stardew Valley. "
            f"They live in {row['Lives In']} at {row['Address']}. "
            f"Marital Status: {'Eligible for marriage' if str(row['Marriage']).lower() == 'yes' else 'Not eligible'}. "
            f"Birthday: {row['Birthday Season']} {row['Birthday Day']}. "
            f"Family: {clean_list_str(row['Family'])}. "
            f"Clinic Visits: {row['Clinic Visit']}. "
            f"\nGifts Guide for {name}:"
            f"\n- Loves: {clean_list_str(row['Loved Gifts'])}"
            f"\n- Likes: {clean_list_str(row['Liked Gifts'])}"
            f"\n- Dislikes: {clean_list_str(row['Disliked Gifts'])}"
            f"\n- Hates: {clean_list_str(row['Hated Gifts'])}"
        )
        
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": file_path,
                    "character": name,
                    "location": row['Lives In'],
                    "type": "character_info"
                }
            )
        )
    return docs

def build_index():
    if not os.path.exists(DATA_FILE):
        print(f"HATA: {DATA_FILE} bulunamadı.")
        return 0

    print(f"Processing {DATA_FILE}...")
    all_docs = process_stardew_csv(DATA_FILE)

    # Her karakterin verisi bir blok olduğu için karakter bazlı ayırıyoruz
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=0,
        separators=["\n\n", "\n"]
    )
    
    chunks = splitter.split_documents(all_docs)
    print(f"Total characters processed: {len(all_docs)}. Chunks created: {len(chunks)}")

    # Embeddings ve Vektör Veritabanı
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    vectorstore = Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    # Toplu yükleme
    vectorstore.add_documents(chunks)
    print("✅ Stardew Valley Character Index updated successfully.")
    return len(chunks)

if __name__ == "__main__":
    build_index()
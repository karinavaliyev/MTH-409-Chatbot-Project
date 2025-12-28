import pandas as pd
import os
import ast
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
# PDF İşleyiciler
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- AYARLAR ---
CHROMA_DIR = "./.chroma_stardew"
COLLECTION = "stardew-knowledge"
EMBED_MODEL = "text-embedding-3-large"
PDF_FILE = "data/StardewValley-Guide.pdf" # PDF dosyanın tam adı

# --- YARDIMCI FONKSİYONLAR ---
def clean_list_str(s):
    if pd.isna(s) or s == "": return "Belirtilmemiş"
    try:
        lst = ast.literal_eval(s)
        return ", ".join(lst) if isinstance(lst, list) else str(s)
    except:
        return str(s).replace("[", "").replace("]", "").replace("'", "")

def safe_get(value, suffix=""):
    if pd.isna(value): return "Bilinmiyor"
    return f"{value}{suffix}"

# --- 1. CSV İŞLEME (Karakterler ve Ekinler) ---
def process_csv_data():
    docs = []
    
    # KARAKTERLER
    if os.path.exists("data/characters.csv"):
        print("📄 CSV: Karakterler işleniyor...")
        df_char = pd.read_csv("data/characters.csv")
        for _, row in df_char.iterrows():
            text = (
            f"Karakter Adı: {row["Name"]}.\n"
            f"Yaşadığı Yer: {row['Lives In']} bölgesi, Adres: {row['Address']}.\n"
            f"Doğum Günü: {row['Birthday Season']} mevsiminin {row['Birthday Day']}. günü.\n"
            f"Aile Üyeleri: {clean_list_str(row['Family'])}.\n"
            f"Evlilik Durumu: {'Evlenilebilir' if str(row['Marriage']).lower() == 'yes' else 'Evlenilemez'}.\n"
            f"Klinik Ziyaret Günü: {row['Clinic Visit']}.\n"
            f"HEDİYE REHBERİ:\n"
            f"- Çok Sevdiği (Love): {clean_list_str(row['Loved Gifts'])}\n"
            f"- Sevdiği (Like): {clean_list_str(row['Liked Gifts'])}\n"
            f"- Nötr Olduğu (Neutral): {clean_list_str(row['Neutral Gifts'])}\n"
            f"- Sevmediği (Dislike): {clean_list_str(row['Disliked Gifts'])}\n"
            f"- Nefret Ettiği (Hate): {clean_list_str(row['Hated Gifts'])}"
            )
            docs.append(Document(page_content=text, metadata={"source": "csv_character", "name": row['Name']}))

    # EKİNLER
    if os.path.exists("data/crops.csv"):
        print("🌱 CSV: Ekinler işleniyor...")
        df_crop = pd.read_csv("data/crops.csv")
        for _, row in df_crop.iterrows():
            crop_name = row['Name'] if pd.notna(row['Name']) else str(row['Seed']).replace(" Seeds", "")
            
            # Regrowth (Tekrar büyüme) kontrolü
            regrowth_info = ""
            if pd.notna(row['Regrowth Time (In Days)']):
                regrowth_info = f"Bu bitki hasat edildikten sonra ölmeyip her {int(row['Regrowth Time (In Days)'])} günde bir tekrar ürün verir."
            else:
                regrowth_info = "Bu bitki tek hasatlıktır, hasattan sonra tekrar ekilmesi gerekir."
            
            text = (
            f"Ekin Adı: {crop_name}.\n"
            f"Yetiştiği Mevsimler: {clean_list_str(row['Season'])}.\n"
            f"Tohum Bilgisi: '{row['Seed']}' olarak geçer.\n"
            f"Tohum Satın Alma Yerleri: {clean_list_str(row['Purchase Source'])}.\n"
            f"Tohum Satış Fiyatı (Oyuncunun satışı): {safe_get(row['Sell Price (Seed)'], ' altın')}.\n"
            f"Büyüme Süresi: Toplam {row['Growth Time (In Days)']} gün sürer.\n"
            f"Hasat Özelliği: {regrowth_info}\n"
            f"Hasat Başına Kazanılan XP: {safe_get(row['XP'])} puan.\n"
            f"ÜRÜN SATIŞ FİYATLARI (Kaliteye Göre):\n"
            f"- Normal Kalite: {row['Price (Regular)']} altın\n"
            f"- Gümüş (Silver) Kalite: {row['Price (Silver)']} altın\n"
            f"- Altın (Gold) Kalite: {row['Price (Gold)']} altın\n"
            f"- İridyum (Iridium) Kalite: {row['Price (Iridium)']} altın"
            )
            docs.append(Document(page_content=text, metadata={"source": "csv_crop", "name": str(crop_name)}))
            
    return docs

# --- 2. PDF İŞLEME (Genel Rehber) ---
def process_pdf_guide():
    if not os.path.exists(PDF_FILE):
        print(f"UYARI: {PDF_FILE} bulunamadı, PDF işlenmeyecek.")
        return []

    print("📚 PDF: Rehber kitabı okunuyor ve parçalanıyor (Bu biraz sürebilir)...")
    
    loader = PyPDFLoader(PDF_FILE)
    raw_pages = loader.load()
    
    # PDF çok uzun olduğu için küçük parçalara bölüyoruz
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Her parça 1000 karakter
        chunk_overlap=200,    # Parçalar birbirine bağlansın diye 200 karakter tekrar etsin
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_pages)
    
    # Metadata düzenleme (Her parça hangi sayfadan geldi bilinsin)
    for doc in chunks:
        doc.metadata["source"] = "pdf_guide"
        # Sayfa numarasını alıp isme ekliyoruz
        page_num = doc.metadata.get("page", 0) + 1
        doc.metadata["name"] = f"Rehber Kitap (Sayfa {page_num})"
        
    print(f"📚 PDF İşlendi: {len(chunks)} parça oluşturuldu.")
    return chunks

# --- ANA ÇALIŞTIRMA ---
def main():
    all_documents = []
    
    # CSV'leri al
    all_documents.extend(process_csv_data())
    
    # PDF'i al
    all_documents.extend(process_pdf_guide())

    if not all_documents:
        print("❌ Hiçbir veri bulunamadı!")
        return

    print(f"\n💾 Toplam {len(all_documents)} bilgi parçasını veritabanına yüklüyorum...")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    # Mevcut veritabanını sıfırdan oluşturmak daha sağlıklı (üst üste binmemesi için)
    if os.path.exists(CHROMA_DIR):
        print("⚠️ Eski veritabanı üzerine yazılıyor...")
        
    vectorstore = Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    
    # Hata almamak için 50'şerli gruplar halinde yükle
    batch_size = 50
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i+batch_size]
        vectorstore.add_documents(batch)
        print(f"✅ {i + len(batch)} / {len(all_documents)} yüklendi.")

    print("\n🎉 TEBRİKLER! Artık chatbot'un hem CSV hem PDF verilerine hakim.")

if __name__ == "__main__":
    main()
import pandas as pd
import os
import ast
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Ortam değişkenlerini yükle
load_dotenv()

# --- AYARLAR ---
CHROMA_DIR = "./.chroma_stardew"
COLLECTION = "stardew-knowledge"  # Karakterler ve ekinler tek yerde
EMBED_MODEL = "text-embedding-3-large"

# --- YARDIMCI FONKSİYONLAR ---
def clean_list_str(s):
    """['Elma', 'Armut'] şeklindeki stringleri temiz, okunur listeye çevirir."""
    if pd.isna(s) or s == "":
        return "Belirtilmemiş"
    try:
        # String listeyi gerçek listeye çevir
        lst = ast.literal_eval(s)
        if isinstance(lst, list):
            if not lst: return "Hiçbiri"
            return ", ".join(lst)
        return str(s)
    except:
        # Eğer liste formatında değilse, gereksiz karakterleri temizle
        return str(s).replace("[", "").replace("]", "").replace("'", "").replace('"', "")

def safe_get(value, suffix=""):
    """Değer boşsa 'Bilinmiyor' döner, değilse değeri ve eki döner."""
    if pd.isna(value):
        return "Bilinmiyor"
    return f"{value}{suffix}"

# --- 1. KARAKTER VERİSİNİ İŞLEME (Tüm Sütunlar) ---
def process_characters(file_path):
    docs = []
    df = pd.read_csv(file_path)
    
    print(f"📄 Karakter verisi işleniyor ({len(df)} satır)...")
    
    for _, row in df.iterrows():
        name = row['Name']
        
        # Tüm sütunları kapsayan detaylı metin
        text = (
            f"Karakter Adı: {name}.\n"
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
        
        docs.append(Document(
            page_content=text, 
            metadata={"source": "characters", "name": name}
        ))
    return docs

# --- 2. EKİN VERİSİNİ İŞLEME (Tüm Sütunlar: Fiyatlar, XP, Büyüme vb.) ---
def process_crops(file_path):
    docs = []
    df = pd.read_csv(file_path)
    
    print(f"🌱 Ekin verisi işleniyor ({len(df)} satır)...")
    
    for _, row in df.iterrows():
        # Bazen isim boş olabilir, tohum adından çıkarım yap
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
        
        docs.append(Document(
            page_content=text, 
            metadata={"source": "crops", "name": str(crop_name)}
        ))
    return docs

# --- ANA ÇALIŞTIRMA ---
def main():
    all_documents = []
    
    # Dosyaların varlığını kontrol et ve işle
    if os.path.exists("data/characters.csv"):
        all_documents.extend(process_characters("data/characters.csv"))
    else:
        print("UYARI: characters.csv bulunamadı.")
        
    if os.path.exists("data/crops.csv"):
        all_documents.extend(process_crops("data/crops.csv"))
    else:
        print("UYARI: crops.csv bulunamadı.")

    if not all_documents:
        print("❌ Hiçbir veri bulunamadı! Lütfen CSV dosyalarını proje klasörüne koyduğundan emin ol.")
        return

    print(f"\nToplam {len(all_documents)} adet zenginleştirilmiş veri bloğu oluşturuldu.")
    print(f"Vektör veritabanı ({CHROMA_DIR}) güncelleniyor, lütfen bekleyin...")

    # Embeddings modelini başlat
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    # ChromaDB'yi oluştur veya güncelle
    vectorstore = Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    
    # Verileri parça parça ekle (Hata almamak için)
    batch_size = 50
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i+batch_size]
        vectorstore.add_documents(batch)
        print(f"✅ {i + len(batch)} / {len(all_documents)} veri işlendi.")

    print("\n🎉 İŞLEM TAMAMLANDI! Tüm veriler başarıyla yüklendi.")

if __name__ == "__main__":
    main()
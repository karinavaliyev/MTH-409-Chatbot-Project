import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# RAGAS Kütüphaneleri
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall

load_dotenv()

# --- AYARLAR ---
CHROMA_DIR = "./.chroma_stardew"
COLLECTION = "stardew-knowledge"
OPENAI_MODEL = "gpt-4o" 

# --- 1. SİSTEMİ HAZIRLA ---
print("\n" + "="*30)
print("⚙️  RAG Sistemi Hazırlanıyor...")
print("="*30)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(
    persist_directory=CHROMA_DIR, 
    embedding_function=embeddings,
    collection_name=COLLECTION
)
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# --- 2. TEST VERİ SETİ (Ground Truth) ---
# Hem CSV (net bilgi) hem PDF (yorum/rehber) içeren sorular hazırladım.
# --- HİBRİT TEST VERİ SETİ (CSV + PDF Karışık) ---
test_data = [
    {
        "question": "Altın kalitede bir Karnabahar (Cauliflower) kaç altına satılır?",
        "ground_truth": "Altın kalitede bir Karnabahar 262 altına satılır."
    },
    {
        "question": "Abigail'in doğum günü ne zamandır?",
        "ground_truth": "Abigail'in doğum günü Güz (Fall) mevsiminin 13. günüdür."
    },
    {
        "question": "Kahve Çekirdeği (Coffee Bean) hangi mevsimlerde yetişir?",
        "ground_truth": "Kahve Çekirdeği hem Bahar (Spring) hem de Yaz (Summer) mevsimlerinde yetişir."
    },
    {
        "question": "Ayçiçeği (Sunflower) tohumlarını nereden satın alabilirim?",
        "ground_truth": "Ayçiçeği tohumları Pierre's General Store ve JojaMart'tan satın alınabilir."
    },
    {
        "question": "Maru kimlerle birlikte yaşamaktadır?",
        "ground_truth": "Maru; Robin, Demetrius ve Sebastian ile birlikte yaşamaktadır."
    },
    {
        "question": "Elliot ile evliysen, Pazartesi günleri nereye gider?",
        "ground_truth": "Elliot evliyse Pazartesi günleri sahile (beach) gider."
    },
    {
        "question": "Vincent, Bahar (Spring) ayının 11. günü nereye gider?",
        "ground_truth": "Vincent, Bahar ayının 11. günü kliniğe (clinic) gider."
    },
    {
        "question": "Abandoned Jojamart içindeki 'Missing Bundle'ı tamamlamak için hangi malzemeler gerekir?",
        "ground_truth": "Bu paketi tamamlamak için şu 6 seçenekten 5'i gerekir: Gümüş (Silver) veya üzeri kalitede Şarap, Dinozor Mayonezi (Dinosaur Mayonnaise), Prizmatik Parça (Prismatic Shard), Altın kalite Kadim Meyve (Ancient Fruit), Altın kalite Boşluk Somonu (Void Salmon) ve Havyar (Caviar)."
    },
    {
        "question": "Evlendikten sonra eşlerin (Spouses) çiftlikten ayrılma durumlarına ne denir?",
        "ground_truth": "Bu durumlara 'Marriage Deviations' (Evlilik Sapmaları) denir ve belirli günlerde gerçekleşir."
    },
    {
        "question": "Gece Pazarı (Night Market) hangi mevsimde ve hangi tarihlerde gerçekleşir?",
        "ground_truth": "Gece Pazarı, Kış (Winter) mevsiminde, ayın 15. 16. ve 17. günlerinde gerçekleşir."
    }
]

# --- 3. SORULARI SİSTEME SOR (INFERENCE) ---
print("\n🤖 Sorular sisteme soruluyor...")

questions = []
ground_truths = []
answers = []
contexts = []

for i, item in enumerate(test_data, start=1):
    q = item["question"]
    print(f"   Soru {i}: {q}")

    # A) Retrieval (Arama)
    docs = vectorstore.similarity_search(q, k=5)
    retrieved_texts = [doc.page_content for doc in docs]
    
    # B) Generation (Cevap Üretme)
    context_str = "\n\n".join(retrieved_texts)
    system_msg = f"Sen bir asistansın. Sadece şu bilgilere göre cevap ver:\n{context_str}"
    
    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=q)
    ])
    
    # C) Listelere Ekleme (RAGAS formatı için)
    questions.append(q)
    ground_truths.append(item["ground_truth"])
    answers.append(response.content)
    contexts.append(retrieved_texts) # Dikkat: RAGAS burayı liste içinde liste ister

# --- 4. RAGAS DEĞERLENDİRMESİ ---
print("\n⚖️  RAGAS Hakemi Puanlıyor (Bu işlem biraz sürebilir)...\n")

# Veriyi RAGAS formatına çevir
data_dict = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data_dict)

# Değerlendirmeyi Başlat
# Hocanın özellikle istediği iki metrik: faithfulness ve context_recall
results = evaluate(
    dataset=dataset, 
    metrics=[faithfulness, context_recall]
)

# --- 5. SONUÇLARI RAPORLA ---
print("\n" + "="*30)
print("📊 RAGAS DOĞRULUK RAPORU")
print("="*30)
print(results)

# Detaylı tabloyu Pandas'a çevir
df_results = results.to_pandas()

# Hata ayıklama için sütun isimlerini görelim
print(f"\nTablo Sütunları: {df_results.columns.tolist()}")

# Gösterilecek sütunları dinamik olarak seçelim
cols_to_show = []

# Soru sütununu bul (question veya user_input olabilir)
if 'question' in df_results.columns:
    cols_to_show.append('question')
elif 'user_input' in df_results.columns:
    cols_to_show.append('user_input')

# Metrikleri ekle (Hesaplanmamışsa hata vermesin diye kontrol ediyoruz)
if 'faithfulness' in df_results.columns:
    cols_to_show.append('faithfulness')
if 'context_recall' in df_results.columns:
    cols_to_show.append('context_recall')

# Varsa cevapları da ekleyelim, görmek iyi olur
if 'answer' in df_results.columns:
    cols_to_show.append('answer')

# Tabloyu yazdır
if cols_to_show:
    print("\nDetaylı Skorlar:")
    print(df_results[cols_to_show])
else:
    print("\n⚠️ İstenen sütunlar tabloda bulunamadı. Tüm tablo basılıyor:")
    print(df_results)

# Excel/CSV olarak kaydet
df_results.to_csv("ragas_report.csv", index=False)
print("\nRapor 'ragas_report.csv' olarak kaydedildi.")
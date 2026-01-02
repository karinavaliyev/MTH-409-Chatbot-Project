import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# --- AYARLAR ---
CHROMA_DIR = "./.chroma_stardew"
COLLECTION = "stardew-knowledge"

# answer_relevancy metriğini 1 generation ile yapılandır
answer_relevancy = AnswerRelevancy(strictness=1)

# --- 1. MODELLERİ TANIMLA ---
print("\n" + "="*50)
print("⚙️  RAG Sistemi Hazırlanıyor...")
print("="*50)

models_to_test = [
    {"name": "GPT-4o", "llm": ChatOpenAI(model="gpt-4o", temperature=0)},
    {"name": "gemini-3-flash-preview", "llm": ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)}
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings, collection_name=COLLECTION)

# RAGAS değerlendirmesi için OpenAI LLM ve Embeddings kullanacağız
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

# --- 2. TEST VERİ SETİ (Ground Truth) ---
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
    },
    {
        "question": "Stardew Valley'de nasıl araba sürülür?",
        "ground_truth": "Stardew Valley'de araba sürme özelliği bulunmamaktadır."
    },
    {
        "question": "Oyunun 1.7 güncellemesi ile eklenecek olan yeni gezegenler ve uzay gemisi yakıtı tarifleri nelerdir?",
        "ground_truth": "Ben sadece Stardew Valley hakkında yardımcı olabilirim! 🧑‍🌾 Stardew Valley ile ilgili başka bir sorunuz varsa, sormaktan çekinmeyin!"
    }
]

all_model_results = []

# --- 3. KARŞILAŞTIRMALI TEST DÖNGÜSÜ ---
for model_info in models_to_test:
    model_name = model_info["name"]
    llm = model_info["llm"]
    
    print(f"\n{'='*50}")
    print(f"🚀 {model_name} test ediliyor...")
    print("="*50)
    
    questions, ground_truths, answers, contexts = [], [], [], []

    for i, item in enumerate(test_data, start=1):
        q = item["question"]
        print(f"   Soru {i}: {q[:50]}...")
        
        # Retrieval
        docs = vectorstore.similarity_search(q, k=6)
        retrieved_texts = [doc.page_content for doc in docs]
        
        # Generation
        context_str = "\n\n".join(retrieved_texts)
        system_msg = f"Sen bir asistansın. Sadece şu bilgilere göre cevap ver:\n{context_str}"
        
        response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=q)])
        
        questions.append(q)
        ground_truths.append(item["ground_truth"])
        answers.append(response.content)
        contexts.append(retrieved_texts)

    # RAGAS Değerlendirmesi (RAGAS 0.2+ formatı)
    print(f"\n📊 {model_name} için RAGAS değerlendirmesi yapılıyor...")
    
    ds = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    })
    
    result = evaluate(
        dataset=ds, 
        metrics=[faithfulness, context_recall, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    # Sonucu listeye ekle
    res_df = result.to_pandas()
    res_df["model"] = model_name
    all_model_results.append(res_df)
    
    # Her model için özet göster
    print(f"\n✅ {model_name} Sonuçları:")
    print(f"   Faithfulness:      {res_df['faithfulness'].mean():.4f}")
    print(f"   Context Recall:    {res_df['context_recall'].mean():.4f}")
    print(f"   Answer Relevancy:  {res_df['answer_relevancy'].mean():.4f}")

# --- 4. KARŞILAŞTIRMALI RAPOR ---
final_df = pd.concat(all_model_results, ignore_index=True)

print("\n" + "="*60)
print("📊 MODEL KARŞILAŞTIRMA RAPORU (ÖZET)")
print("="*60)

# Modellere göre ortalama skorları göster
summary = final_df.groupby("model")[['faithfulness', 'context_recall', 'answer_relevancy']].mean()
print("\n" + summary.to_string())

# En iyi modeli belirle
print("\n" + "-"*60)
print("🏆 EN İYİ MODEL ANALİZİ:")
print("-"*60)
for metric in ['faithfulness', 'context_recall', 'answer_relevancy']:
    best_model = summary[metric].idxmax()
    best_score = summary[metric].max()
    print(f"   {metric:20s}: {best_model} ({best_score:.4f})")

# Detaylı raporu kaydet
final_df.to_csv("model_comparison_report.csv", index=False)
print(f"\n💾 Detaylı rapor 'model_comparison_report.csv' olarak kaydedildi.")
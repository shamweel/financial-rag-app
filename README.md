# 📊 Financial RAG Q&A System  
**A Retrieval-Augmented Generation (RAG) system** that answers **finance-related questions** based on company financial reports. The system retrieves relevant **10-K filings**, processes them using **hybrid retrieval (BM25 + FAISS)**, re-ranks with a **Cross-Encoder**, and generates a **fact-checked response** using **LLM Guard**.

**Live Demo:** [Check it out on Streamlit!](https://financial-rag-app-cbawf4kxndavkse7bzcceg.streamlit.app/)

---

## **Features**
✅ **Extracts Financial Data from PDFs**  
✅ **Hybrid Retrieval (BM25 + FAISS) for accuracy**  
✅ **Cross-Encoder for better ranking**  
✅ **LLM Guard for hallucination prevention**  
✅ **Confidence Score for response reliability**  
✅ **Streamlit Web Interface for easy interaction**  

---

## 🔧 **Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/shamweel7/financial-rag-app.git
cd financial-rag-app

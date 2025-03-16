import streamlit as st
import importlib
import rag_pipeline  # Import the pipeline module

# 🚀 Function to Reload the Pipeline
def reload_rag_pipeline():
    try:
        importlib.reload(rag_pipeline)
        st.sidebar.success("RAG pipeline reloaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error reloading pipeline: {e}")

# 🚀 Initialize Streamlit App
st.set_page_config(page_title="Financial Q&A - RAG System", layout="wide")
st.title("📊 Financial Q&A System - Retrieval-Augmented Generation (RAG)")

st.write(
    "🔎 This system answers **finance-related questions** based on company financial reports "
    "from the last two years. **Enter a query below to get insights!**"
)

# 🔄 Sidebar: Reload Button
if st.sidebar.button("Reload RAG System"):
    reload_rag_pipeline()

# 📝 User Query Input
query_input = st.text_input(
    "💬 **Ask a Financial Question:**",
    placeholder="e.g., What was Amazon's net profit in 2023?"
)

# 🚀 Button to Process Query
if st.button("Get Answer"):
    if query_input.strip():  # Ensure non-empty input
        with st.spinner("Processing your query... ⏳"):
            try:
                # Call updated function from the module
                result = rag_pipeline.generate_guarded_response(query_input)

                if isinstance(result, tuple) and len(result) == 2:
                    response, confidence = result  # Ensure correct tuple unpacking
                else:
                    response, confidence = result, None  # Fallback if confidence score isn't returned

                # 🎯 Display AI Response
                st.subheader("📌AI Generated Answer:")
                st.write(f"**{response}**")  # Ensure only the answer is displayed

                # 🔥 Show Confidence Score (Default to 50% if missing)
                if confidence is not None:
                    st.metric(label="Confidence Score", value=f"{confidence:.2f}")
                else:
                    st.metric(label="Confidence Score", value="N/A")

            except ModuleNotFoundError:
                st.error("RAG pipeline module not found. Please ensure it is installed correctly.")
            except AttributeError:
                st.error("Unexpected error: Check if `generate_guarded_response` exists in `rag_pipeline`.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    else:
        st.warning("⚠️ Please enter a valid financial question.")

# ℹ️ Sidebar Information
st.sidebar.title("ℹ️ How It Works")
st.sidebar.markdown(
    """
    - 🔍 **Retrieves relevant financial statements**.
    - 🏆 **Uses hybrid retrieval (BM25 + FAISS) for better accuracy**.
    - 🎯 **Re-ranks results using a Cross-Encoder**.
    - 🛡 **Applies guardrails to ensure factual consistency**.
    - 🤖 **Generates answers using an open-source language model**.
    """
)

# ✅ Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed as part of the AI/ML coursework 🎓")

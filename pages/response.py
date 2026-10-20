
import streamlit as st

from preprocessing import load_data, prepare_data, prepare_documents
from rag import load_embedding_model, encode_texts, retrieve_answer

st.set_page_config(
    page_title="AI Customer Support Assistant",
    layout="wide"
)

st.title("AI Customer Support Assistant")
st.write("Ask a question about a product or customer support.")

@st.cache_resource
def get_model():
    return load_embedding_model()

@st.cache_resource
def get_knowledge_base():
    df = prepare_data(load_data())
    documents = prepare_documents(df)

    tokenizer, model = get_model()
    embeddings = encode_texts(documents, tokenizer, model)

    return df, documents, embeddings

try:
    with st.spinner("Loading knowledge base..."):
        df, documents, embeddings = get_knowledge_base()
        tokenizer, model = get_model()

    st.success(f"Knowledge base ready: {len(documents)} documents")

    question = st.text_input(
        "Your question",
        placeholder="Example: How can I return a product?"
    )

    if st.button("Get answer") and question.strip():

        results = retrieve_answer(
            question,
            documents,
            embeddings,
            tokenizer,
            model,
            top_k=3
        )

        best = results[0]

        st.subheader("Retrieved answer")
        st.write(best["document"])
        st.caption(f"Similarity score: {best['score']:.3f}")

        with st.expander("See other relevant documents"):
            for result in results[1:]:
                st.write(result["document"])
                st.caption(
                    f"Similarity score: {result['score']:.3f}"
                )

except Exception as e:
    st.error(f"An error occurred: {e}")

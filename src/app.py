import streamlit as st

from generate import ask

st.set_page_config(page_title="Paper Q&A", page_icon="📄")
st.title("Ask the papers")
st.caption("Answers come only from the loaded research papers and may be wrong. "
           "Not medical advice.")

with st.sidebar:
    k = st.slider("Passages to retrieve", 2, 8, 4)
    show_all = st.checkbox("Show all retrieved passages", value=False)

question = st.text_input("Your question")

if st.button("Ask") and question.strip():
    with st.spinner("Searching and answering..."):
        try:
            result = ask(question.strip(), k=k)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    if result["found"]:
        st.success(result["answer"])
    else:
        st.warning("Not found in the loaded documents.")

    top = max((r["score"] for r in result["retrieved"]), default=0.0)
    st.caption(f"Top retrieval similarity: {top:.3f}")

    if result["citations"]:
        st.subheader("Sources")
        for c in result["citations"]:
            with st.expander(f"{c['paper']} - page {c['page']} (similarity {c['score']:.3f})"):
                st.write(c["text"])

    if show_all:
        st.subheader("All retrieved passages")
        for r in result["retrieved"]:
            with st.expander(f"{r['paper']} - page {r['page']} (similarity {r['score']:.3f})"):
                st.write(r["text"])
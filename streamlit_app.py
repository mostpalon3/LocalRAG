import os
import sys
import json
from typing import List

import streamlit as st


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rag_pipeline import RAGPipeline
from impl import Datastore, Indexer, Retriever, ResponseGenerator, Evaluator


DEFAULT_SOURCE_PATH = "sample_data/source/"


@st.cache_resource
def create_pipeline() -> RAGPipeline:
    datastore = Datastore()
    indexer = Indexer()
    retriever = Retriever(datastore=datastore)
    response_generator = ResponseGenerator()
    evaluator = Evaluator()
    return RAGPipeline(datastore, indexer, retriever, response_generator, evaluator)


def get_files_in_directory(source_path: str) -> List[str]:
    if os.path.isfile(source_path):
        return [source_path]
    if not os.path.isdir(source_path):
        return []
    return [
        os.path.join(source_path, filename)
        for filename in sorted(os.listdir(source_path))
        if os.path.isfile(os.path.join(source_path, filename))
    ]


def save_uploaded_files(uploaded_files, uploads_dir: str) -> List[str]:
    os.makedirs(uploads_dir, exist_ok=True)
    saved_paths = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(file_path, "wb") as file_handle:
            file_handle.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)

    return saved_paths


st.set_page_config(page_title="LocalRAG Chatbot", page_icon="💬", layout="wide")
st.title("LocalRAG Chatbot")
st.caption("A local RAG chatbot powered by Ollama, LanceDB, and Streamlit.")

pipeline = create_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None


def load_evaluation_items(raw_text: str):
    items = json.loads(raw_text)
    if not isinstance(items, list):
        raise ValueError("Evaluation input must be a JSON list of question/answer objects.")

    normalized_items = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} must be an object with 'question' and 'answer' keys.")

        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()

        if not question or not answer:
            raise ValueError(f"Item {index} must include non-empty 'question' and 'answer' values.")

        normalized_items.append({"question": question, "answer": answer})

    return normalized_items

with st.sidebar:
    st.header("Controls")
    source_path = st.text_input("Documents path", value=DEFAULT_SOURCE_PATH)
    top_note = st.info("Reset the index before re-adding documents if you change embeddings.")

    if st.button("Reset index", use_container_width=True):
        pipeline.reset()
        st.success("Datastore reset.")

    if st.button("Index documents", use_container_width=True):
        document_paths = get_files_in_directory(source_path)
        if not document_paths:
            st.warning("No documents found at the selected path.")
        else:
            with st.spinner("Indexing documents..."):
                pipeline.add_documents(document_paths)
            st.success(f"Indexed {len(document_paths)} document file(s).")

    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "Choose files to add to the index",
        accept_multiple_files=True,
        key="uploaded_documents",
    )

    if st.button("Upload and index", use_container_width=True):
        if not uploaded_files:
            st.warning("Please choose at least one file to upload.")
        else:
            uploads_dir = os.path.join(ROOT_DIR, ".streamlit_uploads")
            saved_paths = save_uploaded_files(uploaded_files, uploads_dir)
            with st.spinner("Indexing uploaded documents..."):
                pipeline.add_documents(saved_paths)
            st.success(f"Indexed {len(saved_paths)} uploaded file(s).")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("Dynamic Evaluation")
    st.caption("Paste a JSON list with question and answer fields, or upload a .json file.")

    eval_upload = st.file_uploader(
        "Upload evaluation JSON",
        type=["json"],
        accept_multiple_files=False,
        key="evaluation_json_upload",
    )

    eval_text = st.text_area(
        "Or paste evaluation JSON",
        value="",
        height=220,
        placeholder='''[
  {"question": "Your question here", "answer": "Expected answer here"}
]''',
    )

    if st.button("Run evaluation", use_container_width=True):
        try:
            if eval_upload is not None:
                raw_eval_text = eval_upload.getvalue().decode("utf-8")
            else:
                raw_eval_text = eval_text

            if not raw_eval_text.strip():
                st.warning("Provide evaluation JSON by uploading a file or pasting it into the text area.")
            else:
                evaluation_items = load_evaluation_items(raw_eval_text)
                with st.spinner("Running evaluation..."):
                    results = pipeline.evaluate(evaluation_items)
                st.session_state.evaluation_results = results
                st.success(f"Completed evaluation for {len(results)} item(s).")
        except Exception as error:
            st.error(f"Evaluation failed: {error}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("Ask a question about the indexed documents")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = pipeline.process_query(user_prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


if st.session_state.evaluation_results:
    st.divider()
    st.subheader("Latest Evaluation Results")
    results = st.session_state.evaluation_results
    score = sum(result.is_correct for result in results)
    st.metric("Score", f"{score}/{len(results)}")

    for result in results:
        status = "✅ Correct" if result.is_correct else "❌ Incorrect"
        with st.expander(f"{status} - {result.question}"):
            st.markdown(f"**Response:** {result.response}")
            st.markdown(f"**Expected answer:** {result.expected_answer}")
            st.markdown(f"**Reasoning:** {result.reasoning}")
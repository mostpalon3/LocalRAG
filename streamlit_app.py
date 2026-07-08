import os
import sys
import json
import zipfile
from pathlib import Path
from typing import List

import streamlit as st

from config import DB_PATH, EMBED_MODEL, LLM_MODEL


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import RAGPipeline
from impl import Datastore, Indexer, Retriever, ResponseGenerator, Evaluator
from util.extract_content import extract_image_text, is_image_file
from util.file_discovery import collect_files


DEFAULT_SOURCE_PATH = str(ROOT_DIR)


def format_relative_path(path_value: str) -> str:
    path = Path(path_value)
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.name


@st.cache_resource
def create_pipeline() -> RAGPipeline:
    datastore = Datastore()
    indexer = Indexer()
    retriever = Retriever(datastore=datastore)
    response_generator = ResponseGenerator()
    evaluator = Evaluator()
    return RAGPipeline(datastore, indexer, retriever, response_generator, evaluator)


def get_files_in_directory(source_path: str) -> List[str]:
    return collect_files(source_path)


def save_uploaded_files(uploaded_files, uploads_dir: str) -> tuple[List[str], int]:
    os.makedirs(uploads_dir, exist_ok=True)
    saved_paths = []
    skipped_count = 0

    for uploaded_file in uploaded_files:
        file_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(file_path, "wb") as file_handle:
            file_handle.write(uploaded_file.getbuffer())

        if file_path.lower().endswith(".zip") and zipfile.is_zipfile(file_path):
            extracted_paths, skipped_in_zip = extract_uploaded_zip(file_path, uploads_dir)
            saved_paths.extend(extracted_paths)
            skipped_count += skipped_in_zip
        else:
            saved_paths.append(file_path)

    return saved_paths, skipped_count


def extract_uploaded_zip(zip_path: str, uploads_dir: str) -> tuple[List[str], int]:
    extracted_paths: List[str] = []
    skipped_count = 0
    archive_name = Path(zip_path).stem
    extract_root = Path(uploads_dir) / archive_name
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            member_path = Path(member.filename)
            target_path = extract_root / member_path
            resolved_target = target_path.resolve()
            resolved_root = extract_root.resolve()

            if resolved_root not in resolved_target.parents and resolved_target != resolved_root:
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source_handle, open(target_path, "wb") as target_handle:
                target_handle.write(source_handle.read())

            if is_image_file(str(target_path)) and not extract_image_text(str(target_path)).strip():
                skipped_count += 1
                continue

            extracted_paths.append(str(target_path))

    return extracted_paths, skipped_count


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

    st.subheader("System Info")
    st.metric("Documents Indexed", pipeline.get_document_count())
    st.metric("Total Chunks", pipeline.get_chunk_count())
    st.info(f"Model: {LLM_MODEL}")
    st.info(f"Embeddings: {EMBED_MODEL}")
    st.info(f"Vector DB: {format_relative_path(DB_PATH)}")

    if st.button("Reset index", use_container_width=True):
        pipeline.reset()
        st.success("Datastore reset.")
        st.rerun()

    if st.button("Index documents", use_container_width=True):
        document_paths = get_files_in_directory(source_path)
        if not document_paths:
            st.warning("No documents found at the selected path.")
        else:
            with st.spinner("Indexing documents..."):
                pipeline.add_documents(document_paths)
            st.success(f"Indexed {len(document_paths)} document file(s).")
            st.rerun()

    st.subheader("Upload documents")
    st.caption("Upload individual files or a .zip codebase; zip archives are unpacked before indexing.")
    uploaded_files = st.file_uploader(
        "Choose files or a .zip codebase to add to the index",
        accept_multiple_files=True,
        key="uploaded_documents",
    )

    if st.button("Upload and index", use_container_width=True):
        if not uploaded_files:
            st.warning("Please choose at least one file or zip archive to upload.")
        else:
            uploads_dir = ROOT_DIR / ".streamlit_uploads"
            with st.spinner("Uploading, unpacking, and indexing documents..."):
                saved_paths, skipped_count = save_uploaded_files(uploaded_files, uploads_dir)
                pipeline.add_documents(saved_paths)
            st.success(f"Indexed {len(saved_paths)} uploaded item(s).")
            if skipped_count:
                st.warning(f"Skipped {skipped_count} unreadable image file(s) from the uploaded zip archive.")
            st.rerun()

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
        placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking..."):
            retrieved_chunks = pipeline.retriever.search(user_prompt)
            for chunk in pipeline.response_generator.stream_response(user_prompt, retrieved_chunks):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

    with st.expander("📄 Sources used"):
        if not retrieved_chunks:
            st.caption("No retrieval results were returned.")
        for i, chunk in enumerate(retrieved_chunks):
            st.markdown(f"**Chunk {i+1}** — {chunk.source}")
            st.caption(chunk.content[:300] + ("..." if len(chunk.content) > 300 else ""))

    st.session_state.messages.append({"role": "assistant", "content": full_response})


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
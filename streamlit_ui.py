import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
BACKEND_DOWN_MSG = "⚠️ 无法连接后端，请先启动: `uvicorn app:app --reload`"

st.set_page_config(page_title="Personal Knowledge Base", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------------
# Session state — avoid re-fetching on every Streamlit re-run
# ---------------------------------------------------------------------------

if "docs" not in st.session_state:
    st.session_state.docs = None       # cached document list
if "messages" not in st.session_state:
    st.session_state.messages = []     # chat history
if "doc_version" not in st.session_state:
    st.session_state.doc_version = 0   # bump to invalidate cache
if "processed_file_key" not in st.session_state:
    st.session_state.processed_file_key = None  # prevent re-processing on re-run


def _fetch_documents():
    """Fetch doc list from backend. Called only when cache is stale."""
    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=20)
        if resp.ok:
            st.session_state.docs = resp.json().get("documents", [])
        else:
            st.session_state.docs = []
    except requests.RequestException:
        st.session_state.docs = None  # None = backend unreachable


def _invalidate_cache():
    """Bump version so next render re-fetches document list."""
    st.session_state.doc_version += 1


def _handle_request_error(err: Exception) -> str:
    if isinstance(err, requests.ConnectionError):
        return BACKEND_DOWN_MSG
    if isinstance(err, requests.ReadTimeout):
        return " 后端响应超时，请稍后重试"
    if isinstance(err, requests.Timeout):
        return " 请求超时，请稍后重试"
    return f"请求失败: {err}"


# ---------------------------------------------------------------------------
# Sidebar — Document Management
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 文档管理")

    # --- Upload ---
    uploaded_file = st.file_uploader(
        "上传文档",
        type=["pdf", "md", "txt"],
        help="支持 PDF、Markdown、TXT 格式",
    )

    if uploaded_file:
        file_key = (uploaded_file.name, uploaded_file.size)
        if file_key != st.session_state.processed_file_key:
            with st.spinner("正在解析和索引..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                try:
                    resp = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
                    if resp.ok:
                        data = resp.json()
                        st.session_state.processed_file_key = file_key
                        st.success(f"✅ {data['file_name']} 已入库")
                        st.caption(
                            f"切片 {data['stored_chunks']} 个 · "
                            f"去除重复 {data['duplicates_removed']} 个 · "
                            f"清洗掉 {data['cleaning_stats']['noise_removed_pct']}% 噪声"
                        )
                        _invalidate_cache()
                        st.rerun()
                    else:
                        st.error(resp.json().get("detail", "上传失败"))
                except requests.RequestException as e:
                    st.error(_handle_request_error(e))
                except Exception as e:
                    st.error(f"上传失败: {e}")

    st.divider()

    # --- Document list (cached) ---
    # Only re-fetch when version changes — not on every Streamlit re-run
    cache_key = f"doc_list_{st.session_state.doc_version}"
    if cache_key not in st.session_state:
        _fetch_documents()
        st.session_state[cache_key] = True

    docs = st.session_state.docs

    if docs is None:
        st.caption(BACKEND_DOWN_MSG)
    elif len(docs) == 0:
        st.caption("暂无文档")
    else:
        st.caption(f"已索引 {len(docs)} 个文档")
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"📄 {doc['file_name']}")
            with col2:
                if st.button("🗑", key=f"del_{doc['file_id']}"):
                    try:
                        requests.delete(
                            f"{API_BASE}/documents/{doc['file_id']}",
                            timeout=15,
                        )
                    except requests.RequestException:
                        pass
                    _invalidate_cache()
                    st.rerun()

    # Manual refresh
    if st.button("🔄 刷新文档列表"):
        _invalidate_cache()
        st.rerun()

# ---------------------------------------------------------------------------
# Main — Chat
# ---------------------------------------------------------------------------

st.title("📚 Personal Knowledge Base")
st.caption("上传你的文档，然后向知识库提问。答案会标注引用来源。")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 引用来源"):
                for src in msg["sources"]:
                    st.caption(
                        f"**{src['file_name']}** · 片段 #{src['chunk_index']}\n\n"
                        f"{src['content_preview']}..."
                    )

# Chat input
if question := st.chat_input("向知识库提问..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("检索中..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/chat",
                    json={"question": question},
                    timeout=60,
                )
                if resp.ok:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    retrieved = data.get("retrieved_chunks", [])

                    st.markdown(answer)

                    if retrieved:
                        with st.expander(f"🔍 检索到的 {len(retrieved)} 个片段"):
                            for chunk in retrieved:
                                st.caption(
                                    f"**{chunk['file_name']}** · 片段 #{chunk['chunk_index']}\n\n"
                                    f"{chunk['content_preview']}..."
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    st.error(resp.json().get("detail", "查询失败"))
            except requests.RequestException as e:
                st.error(_handle_request_error(e))
            except Exception as e:
                st.error(f"查询失败: {e}")

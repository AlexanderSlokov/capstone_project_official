import csv
import os
import socket
import sys
import threading

import streamlit as st
import torch
import unicodedata
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from config.configurations import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from localGPT_app.run_localGPT import load_model
from scripts.env_checking import system_check

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

system_prompt = """Bạn là trợ lý có quyền truy cập vào các tài liệu ngữ cảnh cụ thể. Chỉ trả lời các
câu hỏi bằng tiếng Việt, dựa trên ngữ cảnh được cung cấp. Nếu bạn không thể trả lời dựa trên ngữ cảnh,
hãy thông báo cho người dùng một cách lịch sự. Không sử dụng bất kỳ thông tin bên ngoài nào."""


# ===========================================
# 1. Kiểm tra quyền truy cập (Mật khẩu và IP)
# ===========================================


# Trạng thái nhập mật khẩu đúng hoặc sai
def check_password():
    """Hàm kiểm tra mật khẩu với logic cải tiến."""
    if "password_attempted" not in st.session_state:
        st.session_state["password_attempted"] = False  # Trạng thái lần thử mật khẩu đầu tiên

    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        # Hiển thị ô nhập mật khẩu
        password = st.text_input("Nhập mật khẩu:", type="password")

        # Chỉ hiển thị thông báo sai mật khẩu nếu đã thử ít nhất 1 lần
        if password:
            st.session_state["password_attempted"] = True
            if password == "r":  # Thay bằng mật khẩu thực tế
                st.session_state["password_correct"] = True

        if st.session_state["password_attempted"] and not st.session_state["password_correct"]:
            st.error("Mật khẩu không đúng. Vui lòng thử lại.")

        return False  # Chưa xác thực

    return True  # Đã xác thực


def get_client_ip():
    """Lấy địa chỉ IP của host."""
    try:
        # Trả về địa chỉ IP của host trong mạng LAN
        hostname = socket.gethostname()
        client_ip = socket.gethostbyname(hostname)
        return client_ip
    except Exception as get_client_ip_exeption:
        return f"Unknown IP ({get_client_ip_exeption})"


def check_ip_whitelist():
    """Hàm kiểm tra IP người dùng với logic cải tiến."""
    allowed_ips = ["172.25.224.1", "127.0.0.1"]  # Thêm IP cho phép tại đây
    client_ip = get_client_ip()

    if client_ip not in allowed_ips:
        st.error(f"Truy cập bị từ chối: IP của bạn không được phép truy cập.")
        st.stop()


# Xác thực mật khẩu trước
if not check_password():
    st.stop()

# Kiểm tra quyền truy cập theo IP sau khi nhập đúng mật khẩu
check_ip_whitelist()


# ========================================
# 2. Tạo PromptTemplate cho các loại model
# ========================================

def create_prompt_template(system_prompt_setup=system_prompt, prompt_template_type=None, history=False):
    if prompt_template_type == "llama":
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt_setup + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    elif prompt_template_type == "mistral":
        B_INST, E_INST = "<s>[INST] ", " [/INST]"
        if history:
            prompt_template = (
                B_INST
                + system_prompt_setup
                + """

            Context: {history} \n {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                B_INST
                + system_prompt_setup
                + """

            Context: {context}
            User: {question}"""
                + E_INST
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    elif prompt_template_type == "qwen":
        # Cấu trúc prompt cho Qwen
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_PROMPT = B_SYS + system_prompt_setup + E_SYS
        if history:
            instruction = """
            Context: {history} \n {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            instruction = """
            Context: {context}
            User: {question}"""

            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    else:
        # Default cấu trúc nếu không chọn model cụ thể
        if history:
            prompt_template = (
                system_prompt_setup
                + """

            Context: {history} \n {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt_template = (
                system_prompt_setup
                + """

            Context: {context}
            User: {question}
            Answer:"""
            )
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return (
        prompt,
        memory,
    )


# ======================================
# 3. Các hàm tiện ích chung
# ======================================


# Utility function to initialize Streamlit session components
def initialize_component(key, initializer):
    if key not in st.session_state:
        st.session_state[key] = initializer()
    return st.session_state[key]


# Sidebar contents
def add_vertical_space(amount):
    st.markdown(f"{'' * amount}")


def clean_response(response_text):
    # Loại bỏ các thẻ không mong muốn
    unwanted_tags = ["[OUT]", "[INVISIBLE TEXT]", "<<BEGIN>>", "<<END>>"]
    for tag in unwanted_tags:
        response_text = response_text.replace(tag, "")
    # Xóa các khoảng trắng dư thừa
    response_text = response_text.strip()
    return response_text


# Determine the device type
if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"

# =======================================
# Hàm cache để tải một lần duy nhất
# =======================================
model_lock = threading.Lock()


@st.cache_resource
def load_cached_model():
    llm_model = load_model(  # Đổi tên `llm` thành `llm_model` để tránh nhầm lẫn với biến toàn cục
        device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME
    )
    return llm_model


@st.cache_resource
def initialize_embeddings_and_db():
    # Sử dụng tên rõ ràng hơn cho các biến
    hf_embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE_TYPE},
        embed_instruction="Represent the document content for retrieval in Vietnamese:",
        query_instruction="Represent the query content for retrieval in Vietnamese:",
    )

    chroma_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=hf_embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    return hf_embeddings, chroma_db


@st.cache_resource
def create_cached_retriever(_hf_embeddings, _chroma_db, _method_type, _top_k, _fetch_k, _score_threshold=None,
                            _mmr_lambda=None):
    # Đổi tên `retriever` thành `cached_retriever` để không shadow biến ngoài hàm
    cached_retriever = None  # Đảm bảo khởi tạo biến với giá trị mặc định
    if _method_type == "similarity":
        cached_retriever = _chroma_db.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": _top_k,
                "fetch_k": _fetch_k,
                "score_threshold": _score_threshold,
            },
        )
    elif _method_type == "mmr":
        cached_retriever = _chroma_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": _top_k,
                "fetch_k": _fetch_k,
                "lambda": _mmr_lambda,
            },
        )
    return cached_retriever


@st.cache_data
@st.cache_data
def perform_environment_check():
    """Thực hiện kiểm tra môi trường thông qua system_check và lưu kết quả."""
    # Gọi đến system_check() để thực hiện tất cả các kiểm tra
    results = system_check()
    return results


# ============================================
# 4. Phần chính của giao diện ứng dụng Streamlit
# ============================================


# Sidebar bên cạnh trái.
# Sidebar for system environment checks
with st.sidebar:
    st.title("🤗💬 Trợ lý truy vấn văn bản của bạn.")
    st.subheader("🔍 Kiểm tra môi trường...")

    # Kiểm tra hệ thống nếu chưa có kết quả trong session_state
    if "env_results" not in st.session_state:
        st.session_state["env_results"] = system_check()

    env_results = st.session_state["env_results"]  # Lấy kết quả đã lưu

    # Hiển thị thông tin môi trường
    if env_results:
        st.write(f"**CUDA khả dụng:** {'Có' if env_results.get('CUDA Available') else 'Không'}")
        st.write(f"**Phiên bản CUDA:** {env_results.get('CUDA Version', 'Không xác định')}")

        st.write(f"**VRAM:** {env_results['VRAM']:.2f} GB" if env_results.get("VRAM") else "Không xác định")
        st.write(f"**RAM:** {env_results['RAM']:.2f} GB" if env_results.get('RAM') else "Không xác định")

        st.write(f"**CPU:** {env_results['CPU'].get('CPU', 'Không xác định')}")
        st.write(
            f"**Intel Hyper-Threading:** {'Có' if env_results['CPU'].get('Intel Hyper-Threading') else 'Không'}"
        )
        st.write(f"**CUDA Compute Capability:** {env_results.get('CUDA Capability', 'Không xác định')}")

        st.write(f"**Đường dẫn Conda CUDA:** {env_results.get('Conda Path', 'Không tìm thấy')}")
        st.write(f"**Phiên bản Conda CUDA:** {env_results.get('Conda Version', 'Không tìm thấy')}")
        st.write(f"**NVCC Version:** {env_results.get('NVCC Version', 'Không tìm thấy')}")

        # Hiển thị yêu cầu mô hình
        model_req = env_results.get("Model Requirements", {})
        st.subheader("🧠 Yêu cầu mô hình:")
        if model_req:
            st.write(f"**VRAM cần thiết:** {model_req.get('VRAM Required', 'Không xác định')} GB")
            st.write(f"**RAM cần thiết:** {model_req.get('RAM Required', 'Không xác định')} GB")
            st.write(f"**VRAM đủ:** {'Có' if model_req.get('Sufficient VRAM') else 'Không'}")
            st.write(f"**RAM đủ:** {'Có' if model_req.get('Sufficient RAM') else 'Không'}")
            st.write(f"**Lớp GPU đề xuất:** {model_req.get('Suggested GPU Layers', 'Không xác định')}")
            st.write(f"**Kích thước batch đề xuất:** {model_req.get('Suggested Batch Size', 'Không xác định')}")
        else:
            st.write("Không đủ thông tin để kiểm tra yêu cầu mô hình.")
    else:
        st.error("Không thể thực hiện kiểm tra môi trường!")

    add_vertical_space(5)
    st.write("Ứng dụng này được tạo ra với ❤️ bởi [Prompt Engineer](https://youtube.com/@engineerprompt)")
    st.write("Hoàn thiện và tối ưu dành cho người Việt ️bởi [Đinh Tấn Dũng - Alexander Slokov]("
             "https://github.com/AlexanderSlokov)")
    # st.write("Dựa trên công nghệ của:")
    # st.markdown("- [Streamlit](https://streamlit.io/) - Framework xây dựng ứng dụng web Python dễ dàng.")
    # st.markdown("- [LangChain](https://python.langchain.com/) - Công cụ hỗ trợ xây dựng hệ thống LLM hiệu quả.")
    # st.markdown("- [HuggingFace](https://huggingface.co/) - Cộng đồng phát triển mô hình xử lý ngôn ngữ tiên tiến.")
    # st.markdown("- [ChromaDB](https://www.trychroma.com/) - Bộ máy vector database hiện đại.")
    # st.markdown("- [LocalGPT](https://github.com/PromtEngineer/localGPT) - Khởi nguồn của ứng dụng này.")
    # add_vertical_space(2)
    st.write("Cảm ơn tất cả các công cụ mã nguồn mở và cộng đồng phát triển đã hỗ trợ chúng tôi tạo nên ứng dụng này.")

# Main localGPT_app title
st.title("LocalGPT - Trợ lý truy vấn văn bản AI")

# ============================================
# 4. Tải mô hình và các tài nguyên
# ============================================
try:
    # Load model and embeddings/database
    st.session_state["llm"] = load_cached_model()
    st.success("Mô hình đã được tải và sẵn sàng sử dụng!")

    # Initialize embeddings and database
    if "embeddings" not in st.session_state or "db" not in st.session_state:
        embeddings, db = initialize_embeddings_and_db()
        st.session_state["embeddings"] = embeddings
        st.session_state["db"] = db

    # Giao diện chọn phương pháp truy vấn
    retrieval_method = st.radio(
        "Chọn phương pháp truy vấn:",
        ["similarity - tìm thông tin tương tự", "mmr - tìm thông tin liên quan"],
    )
    method_type = retrieval_method.split(" - ")[0]

    top_k = st.slider(
        "Số lượng tài liệu tương tự được trả về (top_k):",
        min_value=1,
        max_value=50,
        value=20,
        step=1,
    )

    fetch_k = st.slider(
        "Phạm vi tìm kiếm (fetch_k):",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
    )

    retriever = None
    if method_type == "similarity":
        input_score_threshold = st.slider(
            "Ngưỡng điểm tương tự (score_threshold):",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
        )
        retriever = create_cached_retriever(
            _hf_embeddings=st.session_state["embeddings"],  # Đổi tên tham số
            _chroma_db=st.session_state["db"],
            _method_type=method_type,
            _top_k=top_k,
            _fetch_k=fetch_k,
            _score_threshold=input_score_threshold,
        )
    elif method_type == "mmr":
        mmr_lambda = st.slider(
            "Trọng số lambda (MMR):",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )
        retriever = create_cached_retriever(
            _hf_embeddings=st.session_state["embeddings"],  # Đổi tên tham số
            _chroma_db=st.session_state["db"],
            _method_type=method_type,
            _top_k=top_k,
            _fetch_k=fetch_k,
            _mmr_lambda=mmr_lambda,
        )

    # Khởi tạo QA Chain nếu chưa có
    if "QA" not in st.session_state or st.session_state["retriever"] != retriever:
        prompt, memory = create_prompt_template(
            system_prompt_setup=system_prompt, prompt_template_type="qwen", history=False
        )
        st.session_state["QA"] = RetrievalQA.from_chain_type(
            llm=st.session_state["llm"],
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
        st.session_state["retriever"] = retriever
        st.success("QA Chain đã được khởi tạo!")
except Exception as e:
    st.error(f"Lỗi khi tải mô hình hoặc khởi tạo QA Chain: {str(e)}")
    st.stop()

# ==========================================
# 5. Xử lý đầu vào và hiển thị kết quả
# ==========================================
user_query = st.text_input("Nhập câu hỏi của bạn ở đây", key="user_query")

additional_keywords = st.text_input(
    "Thêm từ khoá (keywords) để hệ thống truy vấn có thêm dữ kiện và tìm kiếm chính xác hơn (ngăn cách bởi dấu phẩy, "
    "tuỳ chọn thêm.)",
    key="additional_keywords",
)

if st.button("Gửi câu hỏi"):
    if user_query.strip():
        try:
            # Xử lý từ khóa bổ sung (nếu có)
            if additional_keywords.strip():
                keywords = [kw.strip() for kw in additional_keywords.split(",")]
                enhanced_query = user_query + " " + " ".join(keywords)
            else:
                enhanced_query = user_query

            # Gọi QA với truy vấn đã được nâng cấp
            with st.spinner("Đang xử lý câu hỏi của bạn..."):
                response = st.session_state["QA"](enhanced_query)

            # Hiển thị kết quả
            answer, docs = response["result"], response["source_documents"]
            st.write("### Câu trả lời:")
            st.write(answer)

            # Hiển thị tài liệu tham khảo
            with st.expander("Tài liệu tham khảo:"):
                for i, doc in enumerate(docs):
                    st.write(f"**Tài liệu #{i + 1}:**")
                    st.write(doc.page_content)

            # Lưu lịch sử trò chuyện
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []
            st.session_state["chat_history"].append({
                "question": user_query,
                "answer": answer,
                "documents": docs
            })

        except Exception as e:
            st.error(f"Có lỗi khi xử lý câu hỏi: {str(e)}")
    else:
        st.warning("Vui lòng nhập câu hỏi trước khi gửi!")


# =========================================
# 7. Xuất lịch sử trò chuyện thành CSV
# =========================================
def clean_text(text):
    """
    Loại bỏ các ký tự không hợp lệ và thay thế khoảng trắng không chuẩn bằng khoảng trắng thông thường.
    """
    cleaned_text = []
    for idx, char in enumerate(text):
        # Kiểm tra nếu ký tự thuộc dạng khoảng trắng không chuẩn
        if char.isspace() or unicodedata.category(char) == 'Zs':
            cleaned_text.append(' ')  # Thay bằng khoảng trắng thông thường
        elif char.isprintable() and unicodedata.category(char)[0] not in ['C']:
            cleaned_text.append(char)  # Giữ lại ký tự hợp lệ
        else:
            print(f"Ký tự không hợp lệ tại vị trí {idx}: {repr(char)}")
            cleaned_text.append(' ')  # Thay thế ký tự lỗi bằng khoảng trắng
    return ''.join(cleaned_text)


if st.button("Xuất lịch sử trò chuyện thành CSV"):
    if "chat_history" in st.session_state:
        try:
            # Chuẩn hóa dữ liệu
            raw_data = st.session_state["chat_history"]
            formatted_chat_history = [
                (clean_text(item["question"]), clean_text(item["answer"]))
                for item in raw_data
                if "question" in item and "answer" in item
            ]

            # Kiểm tra nếu danh sách trống
            if not formatted_chat_history:
                raise ValueError("Dữ liệu không đúng định dạng. Không thể xuất CSV.")

            # Ghi vào CSV
            log_dir, log_file = "local_chat_history", "qa_log.csv"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            csv_path = os.path.join(log_dir, log_file)

            with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["Câu hỏi", "Câu trả lời"])  # Tiêu đề
                writer.writerows(formatted_chat_history)  # Dữ liệu

            # Hiển thị thông báo và nút tải xuống
            st.success(f"Lịch sử trò chuyện đã được xuất ra CSV: {csv_path}")
            with open(csv_path, "rb") as file:
                st.download_button(
                    label="Tải xuống CSV",
                    data=file,
                    file_name="chat_history.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Lỗi khi xuất CSV: {str(e)}")
    else:
        st.error("Không có lịch sử trò chuyện để xuất.")

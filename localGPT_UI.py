import datetime
import os
import socket
import sys

import streamlit as st
import torch
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

from config.configurations import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from localGPT_app.run_localGPT import load_model
from localGPT_app.utils import export_to_pdf
from scripts.env_checking import system_check

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

system_prompt = """You are a knowledgeable assistant with access to specific context documents. You must answer the
questions only in Vietnamese language. You must answer questions based on the provided context only.
If you cannot answer based on the context, inform the user politely. Do not use any external information."""

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
            if password == "secure_password":  # Thay bằng mật khẩu thực tế
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

# ============================================
# 4. Phần chính của giao diện ứng dụng Streamlit
# ============================================


# Sidebar bên cạnh trái.
with st.sidebar:
    st.title("🤗💬 Trợ lý truy vấn văn bản của bạn. ")
    st.title("Bảo mật và riêng tư, hoàn toàn nội bộ.")

    # Hiển thị kết quả kiểm tra môi trường
    st.subheader("🔍 Kiểm tra môi trường...")

    # Kiểm tra nếu `env_results` đã tồn tại trong session_state
    if "env_results" not in st.session_state:
        st.session_state["env_results"] = system_check()  # Lưu kết quả vào session_state

    # Lấy kết quả từ session_state
    env_results = st.session_state["env_results"]

    if env_results:
        cuda_available, total_vram, cuda_version = env_results
        st.write(f"CUDA khả dụng: {'Có' if cuda_available else 'Không'}")
        st.write(f"Tổng dung lượng VRAM: {total_vram:.2f} GB" if total_vram else "Không thể lấy thông tin VRAM")
        st.write(f"Phiên bản CUDA: {cuda_version}")
    else:
        st.error("Không thể thực hiện kiểm tra hệ thống!")

    st.markdown(
        """
        ## About
        Ứng dụng này là một LLM-powered chatbot được xây dựng trên nền tảng của:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [LocalGPT](https://github.com/PromtEngineer/localGPT)
        """
    )

    add_vertical_space(5)
    st.write("Ứng dụng này được tạo ra với ❤️ bởi [Prompt Engineer](https://youtube.com/@engineerprompt)")
    st.write("Hoàn thiện và tối ưu dành cho người Việt ️bởi [Đinh Tấn Dũng - Alexander Slokov]("
             "https://github.com/AlexanderSlokov)")
    st.write("Dựa trên công nghệ của:")
    st.markdown("- [Streamlit](https://streamlit.io/) - Framework xây dựng ứng dụng web Python dễ dàng.")
    st.markdown("- [LangChain](https://python.langchain.com/) - Công cụ hỗ trợ xây dựng hệ thống LLM hiệu quả.")
    st.markdown("- [HuggingFace](https://huggingface.co/) - Cộng đồng phát triển mô hình xử lý ngôn ngữ tiên tiến.")
    st.markdown("- [ChromaDB](https://www.trychroma.com/) - Bộ máy vector database hiện đại.")
    st.markdown("- [LocalGPT](https://github.com/PromtEngineer/localGPT) - Khởi nguồn của ứng dụng này.")
    add_vertical_space(2)
    st.write("Cảm ơn tất cả các công cụ mã nguồn mở và cộng đồng phát triển đã hỗ trợ chúng tôi tạo nên ứng dụng này.")

# Main localGPT_app title
st.title("LocalGPT - Trợ lý truy vấn văn bản AI")

# Text input for user query
user_query = st.text_input("Nhập câu hỏi của bạn ở đây", key="user_query")

# Text input for additional keywords
additional_keywords = st.text_input(
    "Thêm từ khoá (keywords) để hệ thống truy vấn có thêm dữ kiện và tìm kiếm chính xác hơn (ngăn cách bởi dấu phẩy, "
    "tuỳ chọn thêm.)",
    key="additional_keywords"
)
# Thêm nút bấm để xác nhận
submit_button = st.button("Gửi câu hỏi")

# =======================================
# 5. Phần chính: Tải mô hình và xử lý câu hỏi
# =======================================


# Thêm checkbox để bật/tắt việc tải mô hình
load_model_flag = st.checkbox("Nạp mô hình AI (Vui lòng bấm chọn để triển khai mô hình AI.)", value=False)

# Kiểm tra trạng thái checkbox trước khi tải mô hình
# Kiểm tra trạng thái checkbox trước khi tải mô hình
if load_model_flag:
    # Khởi tạo trạng thái nếu chưa tồn tại
    if "model_loaded" not in st.session_state:
        st.session_state["model_loaded"] = False
        st.session_state["QA"] = None

    if not st.session_state["model_loaded"]:
        try:
            # Initialize embeddings
            EMBEDDINGS = initialize_component(
                "EMBEDDINGS",
                lambda: HuggingFaceInstructEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    model_kwargs={"device": DEVICE_TYPE},
                    embed_instruction="Represent the document content for retrieval in Vietnamese:",
                    query_instruction="Represent the query content for retrieval in Vietnamese:"
                )
            )

            # Initialize database
            DB = initialize_component(
                "DB",
                lambda: Chroma(
                    persist_directory=PERSIST_DIRECTORY,
                    embedding_function=EMBEDDINGS,
                    client_settings=CHROMA_SETTINGS,
                )
            )

            # Chọn phương pháp truy vấn
            retrieval_method = st.radio(
                "Chọn phương pháp truy vấn:",
                ["similarity - tìm thông tin tương tự.", "mmr - tìm thông tin liên quan."]
            )
            method_type = retrieval_method.split(" - ")[0]

            # Các tham số truy vấn
            top_k = st.number_input(
                "Số lượng tài liệu tương tự sẽ được trả về (k):",
                min_value=1,
                max_value=50,
                value=20,
                step=1
            )
            fetch_k = st.number_input(
                "Phạm vi tìm kiếm bao nhiêu mảnh tài liệu cho câu hỏi của bạn (fetch_k):",
                min_value=10,
                max_value=100,
                value=50,
                step=10
            )

            # Xử lý tùy chọn similarity hoặc mmr
            if method_type == "similarity":
                score_threshold = st.slider(
                    "Ngưỡng điểm tương tự (score_threshold):",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.75,
                    step=0.05
                )
                RETRIEVER = initialize_component(
                    "RETRIEVER",
                    lambda: DB.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": top_k, "fetch_k": fetch_k, "score_threshold": score_threshold}
                    )
                )
            else:
                mmr_lambda = st.slider(
                    "Trọng số lambda:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1
                )
                RETRIEVER = initialize_component(
                    "RETRIEVER",
                    lambda: DB.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": top_k, "fetch_k": fetch_k, "lambda": mmr_lambda}
                    )
                )

            # Initialize LLM
            LLM = initialize_component(
                "LLM",
                lambda: load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
            )

            # Tạo PromptTemplate và Memory
            prompt, memory = create_prompt_template(system_prompt_setup=system_prompt,
                                                    prompt_template_type="qwen", history=False)


            # Tạo QA Chain
            st.session_state["QA"] = RetrievalQA.from_chain_type(
                llm=LLM,
                chain_type="stuff",
                retriever=RETRIEVER,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt, "memory": memory},
            )

            # Đánh dấu mô hình đã được nạp
            st.session_state["model_loaded"] = True
            st.success("Mô hình đã được nạp thành công.")
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi nạp mô hình: {str(e)}")
            st.session_state["model_loaded"] = False
    else:
        st.info("Mô hình đã được nạp trước đó. Không cần nạp lại.")


# ==========================================
# 6. Xử lý đầu vào và xuất kết quả
# ==========================================


# Process user input and display response chỉ khi QA được khởi tạo
if submit_button:
    if not st.session_state.get("model_loaded", False):
        st.error("Mô hình chưa được tải. Vui lòng bật `Load Model` trong sidebar để sử dụng chức năng này.")
    elif st.session_state.get("QA") is None:
        st.error("QA chưa được khởi tạo. Vui lòng kiểm tra quá trình nạp mô hình.")
    elif user_query.strip():
        try:
            # Xử lý từ khóa bổ sung (nếu có)
            if additional_keywords.strip():
                # Tách các từ khóa dựa trên dấu phẩy
                keywords = [kw.strip() for kw in additional_keywords.split(",")]
                # Thêm từ khóa vào truy vấn ban đầu
                enhanced_query = user_query + " " + " ".join(keywords)
            else:
                enhanced_query = user_query

            # Gọi QA với truy vấn được nâng cấp
            with st.spinner("Đang xử lý câu hỏi của bạn..."):
                response = st.session_state["QA"](enhanced_query)  # Lấy QA từ session_state

            answer, docs = response["result"], response["source_documents"]

            cleaned_answer = clean_response(response["result"])
            st.write("### Answer:")
            st.write(cleaned_answer)

            # Expandable section for document similarity search
            with st.expander("Document Similarity Search", expanded=True):
                if docs:
                    for i, doc in enumerate(docs):
                        st.markdown(f"### Source Document #{i + 1}")
                        st.text(doc.metadata.get("source", "Unknown Source"))
                        st.text(doc.page_content)
                else:
                    st.info("No similar documents found.")

            # Lưu lịch sử vào session
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []
            st.session_state["chat_history"].append((user_query, answer))

        except Exception as e:
            st.error(f"Có lỗi xảy ra trong khi xử lý Query: {str(e)}")
    else:
        st.warning("Vui lòng nhập câu hỏi trước khi gửi.")


# =========================================
# 7. Xuất lịch sử trò chuyện thành PDF
# =========================================

if st.button("Xuất lịch sử trò chuyện thành PDF"):
    if "chat_history" in st.session_state and st.session_state["chat_history"]:
        session_id = datetime.datetime  # Tạo ID phiên
        pdf_path = export_to_pdf(session_id, st.session_state["chat_history"])
        st.success(f"Lịch sử trò chuyện đã được xuất ra PDF: {pdf_path}")
        with open(pdf_path, "rb") as file:
            st.download_button(
                label="Tải xuống PDF",
                data=file,
                file_name=f"chat_session_{session_id}.pdf",
                mime="application/pdf"
            )
    else:
        st.error("Không có lịch sử trò chuyện để xuất.")

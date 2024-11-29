import sys
import os

import torch
import streamlit as st
from localGPT_app.run_localGPT import load_model
from langchain.vectorstores import Chroma
from scripts.env_checking import system_check
from config.configurations import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Thêm đường dẫn gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

system_prompt = """You are a knowledgeable assistant with access to specific context documents. You must answer the
questions only in Vietnamese language. You must answer questions based on the provided context only.
If you cannot answer based on the context, inform the user politely. Do not use any external information."""

def model_memory(system_prompt_setup=system_prompt, promptTemplate_type=None, history=False):
    if promptTemplate_type == "llama":
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

    elif promptTemplate_type == "mistral":
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

    elif promptTemplate_type == "qwen":
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



# Utility function to initialize Streamlit session components
def initialize_component(key, initializer):
    if key not in st.session_state:
        st.session_state[key] = initializer()
    return st.session_state[key]


# Sidebar contents
def add_vertical_space(amount):
    st.markdown(f"{'' * amount}")


with st.sidebar:
    st.title("🤗💬 Trợ lý truy vấn văn bản của bạn. Bảo mật và riêng tư, hoàn toàn nội bộ.")
    st.markdown(
        """
        ## About
        Ứng dụng này là một LLM-powered chatbot được xây dựng trên nền tảng của:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [LocalGPT](https://github.com/PromtEngineer/localGPT)
        """
    )

    # Thêm checkbox để bật/tắt việc tải mô hình
    load_model_flag = st.checkbox("Nạp mô hình AI (Vui lòng bấm chọn để triển khai mô hình AI.)", value=False)

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

    add_vertical_space(5)
    st.write("Ứng dụng này được tạo ra với ❤️ bởi [Prompt Engineer](https://youtube.com/@engineerprompt)")
    st.write("Hoàn thiện và tối ưu dành cho người Việt với ❤️ bởi [Đinh Tấn Dũng - Alexander Slokov]("
             "https://github.com/AlexanderSlokov)")
    st.write("Dựa trên công nghệ của:")
    st.markdown("- [Streamlit](https://streamlit.io/) - Framework cho ứng dụng web Python dễ dàng.")
    st.markdown("- [LangChain](https://python.langchain.com/) - Công cụ hỗ trợ xây dựng LLM hiệu quả.")
    st.markdown("- [HuggingFace](https://huggingface.co/) - Mô hình xử lý ngôn ngữ tiên tiến.")
    st.markdown("- [ChromaDB](https://www.trychroma.com/) - Bộ máy vector database hiện đại.")
    st.markdown("- [LocalGPT](https://github.com/PromtEngineer/localGPT) - Khởi nguồn của ứng dụng này.")
    add_vertical_space(2)
    st.write("Cảm ơn tất cả các công cụ mã nguồn mở và cộng đồng phát triển đã hỗ trợ chúng tôi tạo nên ứng dụng này.")

# Determine the device type
if torch.backends.mps.is_available():
    DEVICE_TYPE = "mps"
elif torch.cuda.is_available():
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"
# Kiểm tra trạng thái checkbox trước khi tải mô hình
if load_model_flag:
    # Initialize embeddings
    EMBEDDINGS = initialize_component(
        "EMBEDDINGS",
        lambda: HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
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

    # Initialize retriever
    RETRIEVER = initialize_component("RETRIEVER", lambda: DB.as_retriever())

    # Initialize LLM
    LLM = initialize_component("LLM", lambda: load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID,
                                                         model_basename=MODEL_BASENAME))

    # Sử dụng prompt cho Qwen với lịch sử hội thoại
    prompt, memory = model_memory(promptTemplate_type="qwen", history=False)
    QA = initialize_component(
        "QA",
        lambda: RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=RETRIEVER,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    )

    # Initialize QA pipeline
    prompt, memory = model_memory()
    QA = initialize_component(
        "QA",
        lambda: RetrievalQA.from_chain_type(
            llm=LLM,
            chain_type="stuff",
            retriever=RETRIEVER,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    )
else:
    st.warning("Quá trình khởi tạo mô hình ngôn ngữ đang được tắt để thực hiện kiểm tra môi trường chạy ứng dụng. Vui lòng khởi động quy trình với nút trên để bắt đầu sử dụng.")

# Main localGPT_app title
st.title("LocalGPT - Trợ lý truy vấn văn bản AI")

# Text input for user query
user_query = st.text_input("Nhập câu hỏi của bạn ở đây", key="user_query")

# Text input for additional keywords
additional_keywords = st.text_input(
    "Thêm từ khoá (keywords) (ngăn cách bởi dấu phẩy, tuỳ chọn thêm)", key="additional_keywords"
)
# Thêm nút bấm để xác nhận
submit_button = st.button("Gửi câu hỏi")

# Process user input and display response chỉ khi QA được khởi tạo
if submit_button:
    if QA is None:
        st.error("Mô hình chưa được tải. Vui lòng bật `Load Model` trong sidebar để sử dụng chức năng này.")
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
            response = QA(enhanced_query)
            answer, docs = response["result"], response["source_documents"]

            # Display the answer
            st.write("### Answer:")
            st.write(answer)

            # Expandable section for document similarity search
            with st.expander("Document Similarity Search", expanded=True):
                if docs:
                    for i, doc in enumerate(docs):
                        st.markdown(f"### Source Document #{i + 1}")
                        st.text(doc.metadata.get("source", "Unknown Source"))
                        st.text(doc.page_content)
                else:
                    st.info("No similar documents found.")
        except Exception as e:
            st.error(f"An error occurred while processing the query: {str(e)}")
    else:
        st.warning("Vui lòng nhập câu hỏi trước khi gửi.")

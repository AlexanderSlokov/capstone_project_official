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


# Function to create prompt and memory for QA
def model_memory():
    template = """Use the following pieces of context to answer the question at the end. You must response in
    Vietnamese language. If you don't know the answer, just say that you don't know, don't try to make up an answer.'

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    init_prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    init_memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return init_prompt, init_memory


# Utility function to initialize Streamlit session components
def initialize_component(key, initializer):
    if key not in st.session_state:
        st.session_state[key] = initializer()
    return st.session_state[key]


# Sidebar contents
def add_vertical_space(amount):
    st.markdown(f"{'' * amount}")


with st.sidebar:
    st.title("🤗💬 Chuyển đổi văn bản của bạn ở đây.")
    st.markdown(
        """
        ## About
        Ứng dụng này là một LLM-powered chatbot được xây dựng bởi:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [LocalGPT](https://github.com/PromtEngineer/localGPT)
        """
    )

    # Thêm checkbox để bật/tắt việc tải mô hình
    load_model_flag = st.checkbox("Load Model (Enable for full functionality)", value=False)

    # Hiển thị kết quả kiểm tra môi trường
    st.subheader("🔍 Environment Check")

    # Kiểm tra nếu `env_results` đã tồn tại trong session_state
    if "env_results" not in st.session_state:
        st.session_state["env_results"] = system_check()  # Lưu kết quả vào session_state

    # Lấy kết quả từ session_state
    env_results = st.session_state["env_results"]

    if env_results:
        cuda_available, total_vram, cuda_version = env_results
        st.write(f"CUDA Available: {'Yes' if cuda_available else 'No'}")
        st.write(f"Total VRAM: {total_vram:.2f} GB" if total_vram else "VRAM Info Unavailable")
        st.write(f"CUDA Version: {cuda_version}")
    else:
        st.error("Không thể thực hiện kiểm tra hệ thống!")

    add_vertical_space(5)
    st.write("Được làm với ❤️ bởi [Prompt Engineer](https://youtube.com/@engineerprompt)")
    st.write("Hoàn thiện cho người Việt với ❤️ bởi [Đinh Tấn Dũng - Alexander Slokov]("
             "https://github.com/AlexanderSlokov)")


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
st.title("LocalGPT - Trợ lý đọc văn bản AI")

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

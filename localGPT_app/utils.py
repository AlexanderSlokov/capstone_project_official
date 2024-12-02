import os
import csv
from datetime import datetime
from fpdf import FPDF
from config.configurations import EMBEDDING_MODEL_NAME
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings


def log_to_csv(question, answer):

    log_dir, log_file = "local_chat_history", "qa_log.csv"
    # Ensure log directory exists, create if not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the full file path
    log_path = os.path.join(log_dir, log_file)

    # Check if file exists, if not create and write headers
    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "question", "answer"])

    # Append the log entry
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, question, answer])


def export_to_pdf(session_id, questions_answers):
    """
    Xuất lịch sử cuộc trò chuyện ra file PDF.

    :param session_id: ID phiên trò chuyện.
    :param questions_answers: Danh sách [(câu hỏi, câu trả lời)].
    :return: Đường dẫn tới file PDF đã tạo.
    """
    # Đường dẫn file PDF
    output_dir = "local_chat_history"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"chat_session_{session_id}.pdf")

    # Tạo file PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Thêm tiêu đề
    pdf.set_font("Arial", style="B", size=16)
    pdf.cell(200, 10, txt=f"Lịch sử trò chuyện - Session {session_id}", ln=True, align="C")
    pdf.ln(10)

    # Thêm nội dung
    for idx, (question, answer) in enumerate(questions_answers, start=1):
        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, txt=f"Q{idx}: {question}")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"A{idx}: {answer}")
        pdf.ln(5)

    # Lưu file
    pdf.output(output_path)
    return output_path


def get_embeddings(device_type="cuda"):
    if "instructor" in EMBEDDING_MODEL_NAME:
        return HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            embed_instruction="Represent the document for retrieval:",
            query_instruction="Represent the question for retrieving supporting documents:",
        )

    elif "bge" in EMBEDDING_MODEL_NAME:
        return HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            query_instruction="Represent this sentence for searching relevant passages:",
        )

    else:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type, "trust_remote_code": True},
        )

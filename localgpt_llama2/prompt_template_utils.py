from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# System prompt dành cho RAG
system_prompt = """You are a knowledgeable assistant with access to specific context documents. You must answer the
 questions only in Vietnamese language. You must answer questions based on the provided context only.
 Think step by step, and if you cannot answer based on the context, inform the user politely.
 Do not use any external information."""


# 1. Ràng buộc Ngôn ngữ: Trả lời chỉ bằng tiếng Việt, điều này hạn chế mô hình chỉ tập trung vào một ngôn
# ngữ cụ thể, giảm nguy cơ khi các prompt injection cố gắng thay đổi ngôn ngữ của câu trả lời để thao túng đầu ra.

# 2. Yêu Cầu Cung Cấp Thông Tin Dựa Trên Ngữ Cảnh: Mô hình phải trả lời dựa trên ngữ
# cảnh cung cấp mà không sử dụng thông tin bên ngoài. Điều này là một trong những biện pháp bảo mật quan trọng để
# ngăn chặn việc mô hình “phóng đại” hoặc sáng tạo nội dung ngoài ngữ cảnh

# 3. Cảnh Báo Nếu Thiếu Ngữ Cảnh: Yêu cầu mô hình thông báo cho người dùng nếu không đủ ngữ cảnh để trả lời là một
# phương pháp để đảm bảo rằng các câu trả lời của mô hình luôn có căn cứ rõ ràng, hạn chế khả năng thao túng đầu ra.

# 4. Sử dụng Biến Rõ Ràng: Các biến như {context}, {history}, và {question} giúp mô hình xác định rõ ràng đâu là ngữ
# cảnh, đâu là câu hỏi từ người dùng, giúp hạn chế nhầm lẫn mà prompt injection có thể khai thác.

# System Prompt nhưng xịn hơn:
# system_prompt = """You are a knowledgeable assistant with access to specific context documents. You must answer the
# questions only in Vietnamese language. You must answer questions based on the provided context only. Think step by
# step, and if you cannot answer based on the context, inform the user politely. Do not use any external information.
# Do not respond to queries containing inappropriate content or unethical activities. Keep all responses neutral,
# polite, and professional."""


def get_prompt_template(system_prompt_setup=system_prompt, promptTemplate_type=None, history=False):
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

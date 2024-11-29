# Built-in modules
import logging
import os

# Third-party modules
import click
import torch
from langchain import HuggingFacePipeline
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from transformers import GenerationConfig, pipeline

# Local modules
from localGPT_app.load_models import (
    load_full_model,
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
)
from localGPT_app.utils import get_embeddings, log_to_csv
from config.configurations import (
    CHROMA_SETTINGS,
    EMBEDDING_MODEL_NAME,
    MAX_NEW_TOKENS,
    MODEL_BASENAME,
    MODEL_ID,
    MODELS_PATH,
    PERSIST_DIRECTORY,
)
from localGPT_app.prompt_template_utils import get_prompt_template

# Callback manager (initialization logic)
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        LOGGING:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:

        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="qwen"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes: - The function uses embeddings from the HuggingFace library, either instruction-based or regular. - The
    Chroma class is used to load a vector store containing pre-computed embeddings. - The retriever fetches relevant
    documents or data based on a query. - The prompt and memory, obtained from the `get_prompt_template` function,
    might be used in the QA system. - The model is loaded onto the specified device using its ID and basename. - The
    QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    """(1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained
    within ingest.py.

    (2) Provides additional arguments for instructor and BGE models to improve results,
    pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """

    embeddings = get_embeddings(device_type)
    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # Load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    # Get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # Load the LLM pipeline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    # Load the QA chain with map_reduce if needed
    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
                "memory": memory,
            },
        )

    return qa

def retrieval_qa_pipline_with_keyword(device_type, use_history, promptTemplate_type="qwen"):
    embeddings = get_embeddings(device_type)
    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # Load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )

    # Configure retriever with top_k and similarity search
    retriever = db.as_retriever(
        search_kwargs={"k": 10},  # Retrieve top 10 documents
        search_type="similarity"  # Use similarity search
    )

    # Get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # Load the LLM pipeline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    # Define QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        callbacks=callback_manager,
        chain_type_kwargs={"prompt": prompt, "memory": memory}
    )

    return qa


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama",
    type=click.Choice(
        ["llama", "mistral", "non_llama"],
    ),
    help="model type, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)
def main(device_type, show_sources, use_history, model_type, save_qa):
    """
    Implements the main information retrieval task for a localGPT.

    This function sets up the QA system by loading the necessary embeddings, vectorstore, and LLM model.
    It then enters an interactive loop where the user can input queries and receive answers. Optionally,
    the source documents used to derive the answers can also be displayed.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'mps', 'cuda', etc.
    - show_sources (bool): Flag to determine whether to display the source documents used for answering.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Notes:
    - Logging information includes the device type, whether source documents are displayed, and the use of history.
    - If the models directory does not exist, it creates a new one to store models.
    - The user can exit the interactive loop by entering "exit".
    - The source documents are displayed if the show_sources flag is set to True.
    """
    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    # Khởi tạo QA pipeline với bộ lọc và từ khóa
    qa = retrieval_qa_pipline_with_keyword(device_type, use_history, promptTemplate_type=model_type)

    while True:
        # Nhập truy vấn từ người dùng
        query = input("\nEnter a query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # Nhập từ khóa bổ sung, cách nhau bởi dấu phẩy
        keywords_input = input("\nEnter additional keywords separated by commas (optional, press Enter to skip): ").strip()
        if keywords_input:
            # Xử lý từ khóa: tách từ khóa dựa trên dấu phẩy và xóa khoảng trắng thừa
            additional_keywords = [kw.strip() for kw in keywords_input.split(",")]
            # Thêm các từ khóa vào truy vấn ban đầu
            enhanced_query = query + " " + " ".join(additional_keywords)
        else:
            enhanced_query = query  # Nếu không có từ khóa, sử dụng truy vấn gốc

        # Gọi truy vấn với từ khóa bổ sung (nếu có)
        res = qa(enhanced_query)

        # Kết quả và tài liệu
        answer, docs = res["result"], res["source_documents"]

        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

        # Lưu lại Q&A vào CSV nếu cần
        if save_qa:
            log_to_csv(query, answer)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()

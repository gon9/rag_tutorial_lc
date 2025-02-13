import os
import logging
from dotenv import load_dotenv
import gradio as gr

from langchain_openai.chat_models import ChatOpenAI  # ChatGPT系モデル
from langchain.prompts import PromptTemplate


from extract_text import extract_text_from_pdf
from split_text import split_text
from create_embeddings import create_vectorstore
from chat_qa_chain import create_chat_completion_chain
from rag_qa_chain import create_qa_chain

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .envから環境変数を読み込む
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI APIキーが設定されていません。環境変数 'OPENAI_API_KEY' を設定してください。")

# パス設定
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(SCRIPT_DIR, "..", "data", "aqua_202005.pdf")
VECTORSTORE_PATH = os.path.join(SCRIPT_DIR, "..", "data", "faiss_index")

def load_or_create_vectorstore():
    """既存のベクトルストアがあればロードし、なければ作成する。"""
    if os.path.exists(VECTORSTORE_PATH):
        logger.info("ベクトルストアを読み込み中...")
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        logger.info("ベクトルストアを新規作成中...")
        pdf_text = extract_text_from_pdf(PDF_PATH, start_page=391, end_page=469)
        texts = split_text(pdf_text)
        vectorstore = create_vectorstore(texts, OPENAI_API_KEY)
        vectorstore.save_local(VECTORSTORE_PATH)
        return vectorstore

def chat_interface(query, use_rag, rag_chain, chat_chain):
    """
    use_rag が TrueならRAGモードの回答、
    Falseなら通常のChat Completionモードの回答を行う。
    """
    if use_rag:
        logger.info("RAGモードで回答します。")
        return rag_chain.invoke(query)
    else:
        logger.info("Chat Completionモードで回答します。")
        return chat_chain.invoke(query)

def main():
    vectorstore = load_or_create_vectorstore()
    logger.info("質問応答チェーンを作成中...")
    rag_chain = create_qa_chain(vectorstore)

    logger.info("Chat Completion用のチェーンを作成中...")
    chat_chain = create_chat_completion_chain()

    iface = gr.Interface(
        fn=lambda question, use_rag: chat_interface(question, use_rag, rag_chain, chat_chain),
        inputs=[
            gr.Textbox(lines=2, label="質問を入力"),
            gr.Checkbox(label="RAGモードを使用する")
        ],
        outputs="text",
        title="tutorial"
    )
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()

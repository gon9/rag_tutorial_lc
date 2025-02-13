from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_chat_completion_chain():
    """
    通常のChat Completion用のチェーン。
    ChatOpenAI や LLMChainを使って作成する簡易例。
    """
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",  # 使いたいモデルを指定
        temperature=0.7
    )

    # プロンプトテンプレートを最低限定義
    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "あなたは有能なアシスタントです。以下のユーザーの質問に答えてください。"),
        ("user", "{query}")
        ]
    )

    chat_chain = prompt | llm | StrOutputParser()
    return chat_chain
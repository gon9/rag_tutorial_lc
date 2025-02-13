from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_qa_chain(vectorstore):
    # ChatOpenAIモデルを初期化
    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0
    )
    # リトリーバーの初期化（検索結果の件数を2件に設定）
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )

    prompt = ChatPromptTemplate.from_template('''\
        以下の質問に対して、
        {context}をもとに回答してください。\n\n
        質問: {question}\n回答:
        ''')

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
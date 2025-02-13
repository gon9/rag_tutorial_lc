from langchain_text_splitters import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter でもいいし多分 CharacterTextSplitter でもいい

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
        )
    return text_splitter.split_text(text)


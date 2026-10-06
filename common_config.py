from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
# print(api_key)

model = "gpt-5.4-mini"
temperature = 2
max_tokens = 2086



def llm_connect(
    model: str = model,
    api_key: str = api_key,
    temperature: float = temperature,
    max_tokens: int = max_tokens
):
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=BASE_URL,
        temperature=temperature,
        use_responses_api=False,  # base url로 할 때는 이부분 넣어야 함.(MonoRouter 사용)
        max_tokens=max_tokens,
    )


from langchain_openai import OpenAIEmbeddings

def embedding_model():
    embedding_model = "text-embedding-3-small"
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        base_url=BASE_URL,
        model=embedding_model,
    )
    
    return embeddings

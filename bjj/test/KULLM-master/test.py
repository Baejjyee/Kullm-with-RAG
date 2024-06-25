import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
from utils.prompter import Prompter
#from openai import OpenAI



###
from langchain.document_loaders import  TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
#from duckduckgo_search import DDGS
###


#simpleTextloader -=> raw_document
#simplesplitter -> text_splitter
# class SimpleOpenAIEmbeddings:
#     client = OpenAI()
#     response = client.embeddings.create(input=text,model="text-embedding-ada-002")
#     return response.data[0].embedding
class SimpleTextLoader:
    def __init__(self,file_path):
        self.file_path = file_path
    def load(self):
        text = ''
        with open(self.file_path, 'r',encoding='utf-8') as file:
            text = file.read()
        return  text

class SimpleCharacterTextSplitter:
    def __init__(self,chunk_size,chunk_overlap,separator_pattern='\n\n'):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator_pattern = separator_pattern
    def split_documents(self,documents):
        splits = documents.split(self.separator_pattern)
        chunk = []
        current_chunk = splits[0]

        for split in tqdm(splits[1:],desc="splitting.."):
            if len(current_chunk) +len(split) + len(self.separator_pattern)>self.chunk_size:
                chunk.append(current_chunk.strip())
                current_chunk = split
            else:
                current_chunk += self.separator_pattern
                current_chunk += split
        if current_chunk:
            chunk.append(current_chunk.strip())
        return chunk

class SimpleVectorStore:
    def __init__(self,docs,embedding):
        self.embedding = embedding
        self.documents = []
        self.vectors = []

        for doc in tqdm(docs,desc='embedding...'):
            self.documents.append(doc)
            vector = self.embedding.embed_query(doc)
            self.vectors.append(vector)
    def similarity_search(self,query,k=4):
        query_vector = self.embedding.embed_query(query)
        if not self.vectors:
            return []
        similarities = cosine_similarity([query_vector],self.vectors)[0]
        sorted_doc_similarities = sorted(zip(self.documents,similarities),key=lambda x:x[1],reverse=True)
        return sorted_doc_similarities[:k]

    def as_retrieve(self,k=4):
        return SimpleVectorStore(self,k)
class SimpleRetriver:
    def __init__(self,vector_store,k=4):
        self.vector_store = vector_store
        self.k = k
    def get_relevant_documents(self,query):
        docs = self.vector_store.similarity_search(query, self.k)
        return docs

class SimpleWebSearch:
    def __init__(self,docs=None,embeddings=None):
        self.embeddings = None
        self.docs = []
        self.vectors = []
    def similarity_search(self,query,k=5):
        docs = []
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query,max_results=k)]
        for result in results:
            doc = (result['title']+":"+result['body']+" - "+result['href'],0.0)
            docs.append(doc)
        return docs

    def as_retriver(self,k=4):
        return SimpleRetriver(self,k)#1:17



MODEL = "nlpai-lab/kullm-polyglot-5.8b-v2"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)

prompter = Prompter("kullm")

def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=10000, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result


raw_documents = SimpleTextLoader('raw_of_korea').load()
text_splitter = SimpleCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

embed_model = HuggingFaceEmbeddings(model_name = "jhgan/ko-sbert-sts")
db = SimpleVectorStore(documents,embed_model)

query = "재외국민은 누가 보호해"
docs = db.similarity_search(query)

print("쿼리 출력이 끝났습니다. 질문을 해주세요. 0울 입력하면 종료합니다.")

while True:
    query = input()
    if query == "":
        break
    # docs = db.similarity_search(query)
    # print(docs)
    result = infer(instruction= documents,input_text=query)
    print("구름의 llm을 거쳤습니다. 결과는 다음과 같습니다")
    print(result)


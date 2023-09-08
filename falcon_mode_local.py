from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI

# from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv, find_dotenv
import os

# Load  OpenAI and HuggingFace API keys from .env file and set it as the API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


#  Create a prompt template and Question
question = "Who won the FIFA World Cup in the year 1994? "
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Load the model from HuggingFaceHub
repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 100}
)
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run(question))

# Load the model from OpenAI API
llm_openai = OpenAI(temperature=0.5, max_tokens=64)
llm_chain = LLMChain(prompt=prompt, llm=llm_openai)
print(llm_chain.run(question))

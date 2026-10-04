from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
   

)


model= ChatHuggingFace(llm=llm)

# 1st prompt-> detailed report

template1= PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt- summary


template2= PromptTemplate(
    template='Write a 5 line summary on the following. /n {text}',
    input_variables=['text']
)


prompt1= template1.invoke({'topic':'black hole'})

result= model.invoke(prompt1)

print(result.content)
prompt2= template2.invoke({'text':result.content})



result= model.invoke(prompt2)

print(result.content)

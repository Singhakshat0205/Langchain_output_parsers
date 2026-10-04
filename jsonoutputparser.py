from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv

load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
   

)


model= ChatHuggingFace(llm=llm)

parser= JsonOutputParser()

template= PromptTemplate( 
    template= "Give me name, age and city of a fictional person \n {format_instruction}",

    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}

)

# here langchain is automatically converting template to prompts
chain = template | model | parser
result = chain.invoke({})


print(result)


#with below code we are manually converting out template to prompts 
# prompt = template.format()

# result = model.invoke(prompt)


# print(result)

# final_result= parser.parse(result.content)

# print(final_result['name'])

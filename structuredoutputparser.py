from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser


from dotenv import load_dotenv

load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
   

)


model= ChatHuggingFace(llm=llm)

schema= [

    ResponseSchema(name='fact_1', description= 'Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description= 'Fact 1 about the topic'),
    ResponseSchema(name='fact_3', description= 'Fact 1 about the topic')
]


parser= StructuredOutputParser.from_response_schemas(schema)

template= PromptTemplate(
    template='Give 3 facts about the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)


chain= template | model | parser

# prompt= template.invoke({'topic':'black hole'})


# result= model.invoke(prompt)


# final_result= parser.parse(result.content)

# print(final_result)

result= chain.invoke({'topic':'black hole'})


print(result)
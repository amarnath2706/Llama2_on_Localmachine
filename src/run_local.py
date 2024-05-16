#import libraries
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from src.helper import *

B_INST, E_INST = '[INST]','[/INST]' #instruction token
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n" #system prompt token

#instruction
instruction = "Convert the following text from English to German : \n\n {text}"

#template
#SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS #getting the system prompt
SYSTEM_PROMPT = B_SYS + CUSTOM_SYSTEM_PROMPT + E_SYS #getting the system prompt by adding custom system prompt
template = B_INST + SYSTEM_PROMPT + instruction + E_INST #create the final template

#prompt template
prompt = PromptTemplate(template=template, input_variables=['text'])

#load the llm
llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q4_0.bin',
                    model_type='llama',
                    config={'max_new_tokens':128,
                            'temperature':0.01}
                    )
#LLM Chain
LLM_Chain = LLMChain(prompt=prompt, llm=llm)

print(LLM_Chain.run("How are you?"))
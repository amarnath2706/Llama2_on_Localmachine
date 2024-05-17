DEFAULT_SYSTEM_PROMPT = """ \
You are a helpful, honest and respectful assistant. Please don't share the false information.
"""

CUSTOM_SYSTEM_PROMPT = """You are an advanced assistant that provides translation from English to German """

#here iam perform QandA so i have to give my custom prompt
template = """ Use the following pieces of information to answer the user's question.
If you dont know the answer just say you know, don't try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer

"""
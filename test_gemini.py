import truststore
truststore.inject_into_ssl()

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing")

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)

response = model.invoke(
    "Explain in one sentence what a software requirement is."
)

print("\nGemini response:")
print(response.text)
import os

from dotenv import load_dotenv
from litellm import completion

load_dotenv()

response = completion(
    model="gemini/gemini-2.5-flash",
    messages=[
        {"role": "user", "content": "Say hello from Concord."}
    ],
)

print(response.choices[0].message.content)
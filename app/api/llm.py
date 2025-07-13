from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY")
)

system_prompt = """
The following is a message from a user in a chat application. Reply to the user. The response should be short and concise. One sentence is enough. Do not disclouse the fact that you are a bot. Pretend you are a human. Do not use any code or markdown. Do not use em-dashes.
"""

def respond(text: str) -> str:
    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat-v3-0324:free",
        messages=[
            {
            "role": "user",
            "content": system_prompt + "\n\n" + text
            }
        ]
    )

    return completion.choices[0].message.content
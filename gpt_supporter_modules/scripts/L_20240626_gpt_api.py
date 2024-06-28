from openai import OpenAI
import os

client = OpenAI(api_key="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")

completion = client.chat.completions.create(
  model="gpt-4o",  # モデルの指定
  messages=[
    # {"role": "system", "content": "You are an excellent secretary who responds in Japanese."},
    {"role": "user", "content": "This is the test of API. Please response with a short sentence."}
  ]
)

print(completion.choices[0].message.content)
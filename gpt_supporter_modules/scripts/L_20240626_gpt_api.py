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

print(completion)
# print(completion.choices[0].message.content)
"""
ChatCompletion(
    id='chatcmpl-9f29HvnM6TYS4yu4gMZjHvc7Vb9Bn', 
    choices=[
        Choice(
            finish_reason='stop', 
            index=0, 
            logprobs=None, 
            message=ChatCompletionMessage(
                content='Sure, the API test is successful!', 
                role='assistant', 
                function_call=None, 
                tool_calls=None
                )
                )
            ], 
            created=1719566735, 
            model='gpt-4o-2024-05-13', 
            object='chat.completion', 
            service_tier=None, 
            system_fingerprint='fp_4008e3b719',
            usage=CompletionUsage(
                completion_tokens=8, 
                prompt_tokens=21, 
                total_tokens=29
                )
              )
"""
import google.generativeai as genai
from gtts import gTTS
import tensorflow

genai.configure(api_key="api-key")
cofig = {"max_output_tokens": 1000}
model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=cofig)


# nn = input("Ngôn ngữ (vi, en, vv...): ")
text = input("Bạn muốn hỏi gì: ")

response = model.generate_content(text)
txt = response.text
print(txt)

import google.generativeai as genai

genai.configure(api_key="AIzaSyB3P2COlotMu-3RR-ehwZXZk60wOWJvfEA")
# Liệt kê các model khả dụng
for m in genai.list_models():
    print(m.name)

# Dùng model Gemini mới (ví dụ: gemini-1.5-pro)
model = genai.GenerativeModel('gemini-2.0-pro-exp')
response = model.generate_content("Hello, how are you?")
print(response.text)

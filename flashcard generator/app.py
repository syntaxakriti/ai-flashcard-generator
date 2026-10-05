import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()


class FlashcardGenerator:

    def __init__(self):
        self.client = InferenceClient(
            provider="auto",
            api_key=os.getenv("HF_TOKEN")
        )

    def generate(self, topic):

        prompt = f"""
Create 5 flashcards based on the following topic:

{topic}

Each flashcard should have:
Front: a question
Back: a short and clear answer

Cover different important points.
Do not repeat questions.

Use this format:

Flashcard 1
Front: ...
Back: ...

Flashcard 2
Front: ...
Back: ...

Flashcard 3
Front: ...
Back: ...

Flashcard 4
Front: ...
Back: ...

Flashcard 5
Front: ...
Back: ...
"""

        response = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500
        )

        return response.choices[0].message.content


topic = input("Enter the topic: ")

generator = FlashcardGenerator()

flashcards = generator.generate(topic)

print("\nFlashcards:\n")
print(flashcards)
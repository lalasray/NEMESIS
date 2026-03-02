from openai import OpenAI

def load_api_key(filepath="apikey.txt"):
    with open(filepath, "r") as f:
        return f.read().strip()

def ask_question(question):
    api_key = load_api_key()
    
    # Create client using key from file
    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=question
    )

    return response.output_text


if __name__ == "__main__":
    user_question = input("Enter your question: ")
    answer = ask_question(user_question)
    print("\nAnswer:\n")
    print(answer)
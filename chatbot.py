from dotenv import load_dotenv
import os
import requests
import json


load_dotenv()


class ChatbotSession:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", max_memory=5):
        self.max_memory = max_memory  # Limit memory size (for efficiency)
        self.history = []  # Stores past messages
        self.API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
        hf_token = os.getenv("HF_TOKEN_READ")
        self.__HEADERS = {"Authorization": f"Bearer {hf_token}"}

    def add_to_memory(self, user_msg, bot_reply):
        self.history.append(f"USER: {user_msg}\nBOT: {bot_reply}")
        if len(self.history) > self.max_memory:
            self.history.pop(0)  # Remove oldest conversation

    def get_history(self):
        texts = ""
        for i, chat in enumerate(self.history):
            texts += f"\n\nCHAT : {i+1} : \n"+chat
        return texts

    def generate_prompt(self, context, user_input):
        history = self.get_history()
        prompt = f"""CONTEXT: {context}\n\n
                     According to the context, answer the following question. If user ask something from previous chats, 
                     look into 'PREVIOUS CHATS' section.If the question is out of context, answer "It is out of context!"\n I
                     USER: {user_input}\nBOT:"""
        if history:
            prompt = f"PREVIOUS CHATS\n:{history}\n"+prompt
        return prompt

    def get_response(self, context, user_input):
        prompt = self.generate_prompt(context, user_input)
        response = requests.post(self.API_URL, headers=self.__HEADERS, json={"inputs": prompt})

        # Error handling
        if response.status_code != 200:
            return "Sorry, I couldn't process that request."

        # Parse API response
        bot_reply = response.json()[0]['generated_text'].split("BOT:")[-1].strip()
        self.add_to_memory(user_input, bot_reply)
        return bot_reply


chatbot = ChatbotSession()
if __name__ == '__main__':
    context = """
                Narendra Damodardas Modi[a] (born 17 September 1950)[b] is an Indian politician who has served as the prime minister of India since 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the member of parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindu nationalist paramilitary volunteer organisation. He is the longest-serving prime minister outside the Indian National Congress.[4]

Modi was born and raised in Vadnagar in northeastern Gujarat, where he completed his secondary education. He was introduced to the RSS at the age of eight. At the age of 18, he was married to Jashodaben Modi, whom he abandoned soon after, only publicly acknowledging her four decades later when legally required to do so. Modi became a full-time worker for the RSS in Gujarat in 1971. The RSS assigned him to the BJP in 1985 and he rose through the party hierarchy, becoming general secretary in 1998.[c] In 2001, Modi was appointed chief minister of Gujarat and elected to the legislative assembly soon after. His administration is considered complicit in the 2002 Gujarat riots,[d] and has been criticised for its management of the crisis. According to official records, a little over 1,000 people were killed, three-quarters of whom were Muslim; independent sources estimated 2,000 deaths, mostly Muslim.[13] A Special Investigation Team appointed by the Supreme Court of India in 2012 found no evidence to initiate prosecution proceedings against him.[e] While his policies as chief minister were credited for encouraging economic growth, his administration was criticised for failing to significantly improve health, poverty and education indices in the state.[f]
    """
    print("Chatbot is ready! Type 'exit' to stop.\n")
    while True:
        user_message = input("You: ")
        if user_message.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        bot_response = chatbot.get_response(context,user_message)
        print(f"Chatbot: {bot_response}")

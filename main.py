from langchain_groq import ChatGroq # load the chatmodel
from dotenv import load_dotenv # import  the api key
from langchain_core.prompts import PromptTemplate # reuseable prompt template
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage # role based prompt


load_dotenv() #load the api key



# Model configuration
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.4,
    max_tokens=150
)


chat_history = [SystemMessage(content="You are a helpfull assitant reponce accorinding to query")] # system meesage

while True:
  user_input = input("You:")

  if user_input.lower() in ["exit","bye","stop"]:
    print("Have a nice day!")
    break

  else:

    chat_history.append(HumanMessage(content=user_input))

    result = model.invoke(chat_history)

    chat_history.append(AIMessage(content=result.content))

    print("Coach:",result.content)




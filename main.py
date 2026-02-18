from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import streamlit as st

# load .evv variable (google api key import)
load_dotenv()

# Initialize LLM
model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.35)

# Page configuration in streamlit
st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")
st.header("🤖 LangChain + Streamlit Chatbot")



# create prompt
prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistance and your name is Kishore"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# create chain

chain = prompt | model

# store chat history in steamlit session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

# wrap chain with chat history
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: st.session_state.chat_history,
    input_messages_key="input",
    history_messages_key="history",
)

# display previous message
for msg in st.session_state.chat_history.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# take user input
if user_input := st.chat_input("Type your message...."):
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "streamlit_session"}},
        )
        st.markdown(response.content)

# #check condition user input not be blank
# if st.button("submit"):
#     if user_input.strip() == "":
#         st.warning("Please enter a prompt!")
#     else:
#         st.write("You:", user_input)
#         with st.spinner("Generating response....."):
#             result = model.invoke(user_input)
#             st.subheader("📄 Response:")
#             st.write(result.content)

st.markdown(
    """
    <div style='text-align: center;'>
        <a href="https://github.com/kishoresahoo2050/StreamLitChat" target="_blank">GitHub</a> |
        Built By Kishore Sahoo
    </div>
    """,
    unsafe_allow_html=True,
)

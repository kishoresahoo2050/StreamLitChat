from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

#load .evv variable (google api key import)
load_dotenv()

# Initialize model with gemini
model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.35
)

# set header and title for page
st.set_page_config(page_title="Gemini Research Tool!")
st.header("🔍 AI Research Tool")

#take user input
user_input = st.text_input("Enter your query: "
                           )

#check condition user input not be blank
if st.button("submit"):
    if user_input.strip() == "":
        st.warning("Please enter a prompt!")
    else:
        st.write("You:", user_input)
        with st.spinner("Generating response....."):
            result = model.invoke(user_input)
            st.subheader("📄 Response:")
            st.write(result.content)
            
st.markdown(
    """
    <div style='text-align: center;'>
        <a href="https://github.com/kishoresahoo2050/StreamLitChat" target="_blank">GitHub</a> |
        Built By Kishore Sahoo
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from config import GEMINI_MODEL

from tools import check_missing_appointment_fields, search_documents, parse_date, book_appointment
from document_store import initialize_document_retrieval

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def initialize_agent():
    """Initialize the agent with all required tools."""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    
    tools = [search_documents, parse_date, book_appointment, check_missing_appointment_fields]
    
    agent_prompt = PromptTemplate.from_template("""
    You are an intelligent assistant that helps users find information from Frequently Asked Question document and book appointments.
    You have access to the following tools:

    {tools}

    To use a tool, please use the following format:
    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of {tool_names}
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
    ```
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]
    ```

    When extracting dates, use the parse_date tool.

    For booking appointments, ensure you have:
    - A valid date (YYYY-MM-DD)
    - A time (HH:MM)
    - User's name, email, and phone number
    - Purpose of the appointment

    Only call the book_appointment tool if all required fields are available. Otherwise, ask the user to provide the missing information.

    For callback requests, collect:
    - User's name, email, and phone number
    - Reason for the callback

    When users ask about information that might be in the documents, use the search_documents tool.

    Begin!

    Previous Chat History:
    {chat_history}

    Question: {input}

    {agent_scratchpad}
    """)

    agent = create_react_agent(llm, tools, agent_prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        max_execution_time=3000.0
    )

    return executor

def main():
    st.set_page_config(page_title="AI Document Assistant", page_icon="🤖")
    st.title("AI Document Assistant")

    with st.spinner("Setting things up..."):
        if "vectorstore_initialized" not in st.session_state:
            initialize_document_retrieval()
            st.session_state.vectorstore_initialized = True

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "agent_executor" not in st.session_state:
            st.session_state.agent_executor = initialize_agent()

    if "agent_thinking" not in st.session_state:
        st.session_state.agent_thinking = False

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    disabled_input = st.session_state.agent_thinking
    prompt = st.chat_input("Is there anything I can help you with?", disabled=disabled_input)

    if prompt and not disabled_input:
        st.session_state.agent_thinking = True
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent_executor.invoke({"input": prompt})
                    assistant_response = response.get("output", "I couldn't process that request.")
                except Exception as e:
                    assistant_response = f"Something went wrong: {e}"

            message_placeholder.write(assistant_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        st.session_state.agent_thinking = False

if __name__ == "__main__":
    main()

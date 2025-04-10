# FAQ Chatbot with Appointment Booking

A basic conversational chatbot powered by LangChain, ChromaDB, Streamlit, Pydantic, and the Gemini API. This chatbot can answer user FAQs, naturally extract personal information, and book appointments.

## Features

- Conversational AI chatbot using Gemini API
- Retrieval-augmented generation (RAG) via ChromaDB
- FAQ understanding and response
- Appointment scheduling through natural dialogue
- User input validation using Pydantic
- Simple and interactive UI using Streamlit

## Installation

1. **Clone the repository**


```bash
git clone git@github.com:creatorof/FAQ-Chatbot.git
cd faq-chatbot
```


2. **Create a virtual environment**


```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**


```bash
pip install -r requirements.txt
```

## Configuration


1. **Setting up API key**


Replace the variable value `GEMINI_API_KEY` in the file `secrets.toml` located inside `.streamlit` folder


## Running the chatbot


```bash
streamlit run index.py
```

import importlib
import os
import streamlit as st

class GroqLLM:
    def __init__(self,user_controls_input):
        self.user_controls_input = user_controls_input

    def get_llm_model(self):
        try:
            groq_api_key = self.user_controls_input.get("GROQ_API_KEY", "")
            selected_groq_model = self.user_controls_input.get("selected_groq_model")
            if not groq_api_key and not os.environ.get("GROQ_API_KEY", ""):
                st.error("Please provide a valid GROQ API key.")
                raise ValueError("Missing GROQ API key.")

            groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
            groq_module = importlib.import_module("langchain_groq")
            ChatGroq = getattr(groq_module, "ChatGroq")
            llm = ChatGroq(api_key=groq_api_key, model=selected_groq_model)

        except ModuleNotFoundError:
            raise ValueError("langchain_groq is not installed. Install it to use GroqLLM.")
        except Exception as e:
            raise ValueError(f"Error initializing GroqLLM: {e}")    

        return llm    

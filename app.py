import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

def getLLamaresponse(input_text):
    llm = CTransformers(model='SummarisationModel/llama-2-7b-chat.ggmlv3.q2_K.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})

    template = """
    Feedback: {input_text}    
    Summarize the feedback given above.
    """

    prompt = PromptTemplate(input_variables=["input_text"],
                            template=template)
    
    response = llm.invoke(prompt.format(input_text=input_text))
    
    return response if response else "No response generated"

st.set_page_config(page_title="Summarize Feedback",
                   page_icon='😃', 
                   layout='centered', 
                   initial_sidebar_state='collapsed')

st.header("Generate Summary 😃")

input_text = st.text_input("Enter the feedback")

submit = st.button("Generate")

if submit and input_text:
    response = getLLamaresponse(input_text)
    st.write(response)

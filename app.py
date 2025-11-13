import os
import torch
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*pydantic.*')

# Load environment variables
load_dotenv()

# Set up environment variables from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

#system prompt
system_prompt = """
### System Prompt for Paragraf Lex AI Assistant

Welcome to Paragraf Lex! I am here to help you with any questions you have about VAT and electronic invoicing in Serbia. How can I assist you today?

Role Description:

I am a virtual assistant from Paragraf Lex, specializing in electronic invoicing and Value Added Tax (VAT) legislation in the Republic of Serbia, using information from the Paragraf online legal library. My mission is to provide users with clear, detailed, and accurate information that exceeds previous quality standards.

Response Guidelines:

**Article Integration:**  
I will use relevant sections of the provided articles (segments) related to the user's question.  
I will quote or reference specific sections of laws, articles, or clauses when necessary.

Response Structure:

**1. Brief Introduction:**  
I will confirm my understanding of the question.

**2. Detailed Answer:**  
I will provide comprehensive and easy-to-understand information, referencing the provided articles and regulations.

**3. Legal References:**  
I will cite specific laws, articles, and clauses when relevant.

**4. Conclusion:**  
I will offer additional assistance or clarification if needed.

Error Prevention:

- I will verify the accuracy of all information before providing it.  
- I will avoid making assumptions; if information is missing, I will politely ask for clarification.  
- I will never provide inaccurate or outdated information.

Scope of Response:

**Allowed Topics:**  
Electronic invoicing, VAT, relevant Serbian laws, and related regulations.

**Disallowed Topics:**  
Questions unrelated to electronic invoicing or VAT in Serbia.  
For such queries, I will politely explain this limitation.

Communication Style:

- I will be professional, friendly, and approachable.  
- I will use simple and accessible language suitable for users without legal or accounting backgrounds.  
- I will clearly explain any technical or legal terms.

**Language Consistency:**  
I will always respond **only in English**, regardless of the language used in the user's question.  
If a question is written in another language, I will interpret it but provide the full answer in English.

Article Integration (Segments):

When a user asks a question, the system will provide relevant articles from the Paragraf online legal library as contextual data (segments), which I will use to formulate my response.

Notes:

- I will combine information from the provided data (segments), my own knowledge, and relevant laws to deliver the most accurate answer.  
- I will always consider the latest amendments and updates to Serbian laws and regulations.  
- I will present information as a complete and authoritative response without mentioning internal data sources or segments.

Goal:

My goal is to provide users with the highest quality and most detailed legal information to help them clearly understand and fulfill their obligations related to electronic invoicing and VAT in the Republic of Serbia.
"""


# Initialize OpenAI LLM (with fixed system message)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
index = pc.Index("electronicinvoice1")

# Hugging Face embeddings for text similarity
embedding_function = HuggingFaceEmbeddings(
    model_name="djovak/embedic-base",
    model_kwargs={'device': 'cpu'} if not torch.cuda.is_available() else {'device': 'cuda'}
)

# Pinecone Vectorstore
vectorstore = PineconeVectorStore(index=index, embedding=embedding_function, text_key='text', namespace="text_chunks")

# Retriever for semantic search
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

refinement_template = """Create a focused Serbian search query for the RAG retriever bot. Convert to Serbian language if not already. Include key terms, synonyms, and domain-specific vocabulary. Remove filler words. Output only the refined query in the following format: {{refined_query}},{{keyterms}},{{synonyms}}

Query: {original_question}

Refined Query:"""

refinement_prompt = PromptTemplate(input_variables=["original_question"], template=refinement_template)

# LLM Chain for refinement using LCEL
refinement_chain = refinement_prompt | llm | StrOutputParser()

# Helper function to format documents
def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

# Combined Retrieval Prompt
combined_prompt = ChatPromptTemplate.from_template(
    f"""{system_prompt}

    Please answer the following question using only the context provided:
    {{context}}

    Question: {{question}}
    Answer:"""
)

# Create a retrieval chain using LCEL piping
retrieval_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | combined_prompt
    | llm
    | StrOutputParser()
)

# Processing Query
def process_query(query: str):
    try:
        # Step 1: Refine Query
        refined_query = refinement_chain.invoke({"original_question": query})

        # Step 2: Retrieve and Answer
        response = retrieval_chain.invoke(refined_query)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI - Simplified Version
st.title("Legal Egg AI 🥚")

st.write("Welcome to Serbian E-Invoice and VAT Bot! I'm a comprehensive AI-powered Agent automating VAT compliance, E-invoicing workflows, and regulatory reporting for businesses operating in Serbia. Seamlessly navigate complex tax regulations, ensure precise digital invoicing, and minimize administrative overhead with real-time, intelligent guidance.")

# Sidebar with example questions and clear chat button
with st.sidebar:
    st.header("Common Queries")
    example_questions = [
        "1. When did e-invoicing become mandatory for B2B/B2G transactions in Serbia?",
        "2. What format is required for Serbian e-invoices?",
        "3. How long must e-invoices be stored in Serbia?",
        "4. What penalties apply for non-compliance with e-invoicing rules?",
        "5. Are cross-border transactions subject to Serbian e-invoicing?",
        "6. What are the deadlines for recording input VAT in SEF?",
        "7. Do e-invoices require a digital signature in Serbia?",
        "8. Are there exemptions to Serbia's e-invoicing mandate?",
        "9. How to correct errors in e-invoices or VAT records?",
        "10. How do businesses register for the SEF platform?"
    ]
    for q in example_questions:
        st.markdown(f"• {q}")

    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Manage chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Ask your question..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response
    with st.chat_message("assistant"):
        response = process_query(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})



import streamlit as st
import sqlite3
import json
import os
import re
import PyPDF2
import docx
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.serpapi_tools import SerpApiTools
from phi.tools.website import WebsiteTools
from phi.model.google import Gemini

load_dotenv()
google_api = os.getenv("GOOGLE_API_KEY")
sa = os.getenv("SERPAPI_API_KEY")

genai.configure(api_key=google_api)

def create_pc(api_key):
    return Pinecone(api_key=api_key)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

DB_PATH = "app.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    file_name TEXT,
    pinecone_index TEXT,
    file_content TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    file_id INTEGER,
    role TEXT,
    content TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(file_id) REFERENCES files(id)
)
''')
conn.commit()

def signup(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login(username, password):
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    return user[0] if user else None

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
if 'current_file' not in st.session_state:
    st.session_state['current_file'] = None  
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def show_login_signup():
    st.title("Login / Signup")
    auth_option = st.radio("Select Option", ["Login", "Signup"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if auth_option == "Signup":
        if st.button("Signup"):
            if signup(username, password):
                st.success("Signup successful! Please login.")
            else:
                st.error("Username already exists.")
    else:
        if st.button("Login"):
            user_id = login(username, password)
            if user_id:
                st.session_state['logged_in'] = True
                st.session_state['user_id'] = user_id
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials.")

def handle_file_upload(pc):
    st.subheader("Upload New FAQ File")
    uploaded_file = st.file_uploader("Choose a FAQ file (JSON, TXT, PDF, DOCX)", type=["json", "txt", "pdf", "docx"])
    file_name = st.text_input("Enter a name for this file (used as vector index name)")
    if st.button("Upload File") and uploaded_file and file_name:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        faq_texts = []
        file_content = ""
        try:
            if file_ext == '.json':
                file_content = uploaded_file.read().decode("utf-8")
                faq_data = json.loads(file_content)                
                faq_texts = [f"Q: {item['question']}\nA: {item['answer']}" for item in faq_data]
            elif file_ext == '.txt':
                file_content = uploaded_file.read().decode("utf-8")
                faq_texts = [chunk.strip() for chunk in file_content.split("\n\n") if chunk.strip()]
            elif file_ext == '.pdf':
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                file_content = ""
                for page in pdf_reader.pages:
                    file_content += page.extract_text() + "\n"
                faq_texts = [chunk.strip() for chunk in file_content.split("\n\n") if chunk.strip()]
            elif file_ext == '.docx':
                doc = docx.Document(uploaded_file)
                file_content = "\n".join([para.text for para in doc.paragraphs])
                faq_texts = [chunk.strip() for chunk in file_content.split("\n\n") if chunk.strip()]
            else:
                st.error("Unsupported file type.")
                return
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return
       
        sanitized_name = re.sub('[^a-z0-9-]', '', file_name.strip().lower().replace(" ", "-"))
        file_index = f"faq-{sanitized_name}"
               
        if file_index not in [i['name'] for i in pc.list_indexes()]:
            pc.create_index(
                name=file_index,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1'),
            )
       
        file_index_instance = pc.Index(file_index)
        file_vector_store = PineconeVectorStore(index=file_index_instance, embedding=embeddings)
        file_vector_store.add_texts(faq_texts)
       
        c.execute("INSERT INTO files (user_id, file_name, pinecone_index, file_content) VALUES (?, ?, ?, ?)",
                  (st.session_state['user_id'], file_name, file_index, file_content))
        conn.commit()
        st.success("File uploaded and processed successfully!")

def load_file_for_analysis(file_id):
    c.execute("SELECT id, file_name, pinecone_index, file_content FROM files WHERE id=?", (file_id,))
    record = c.fetchone()
    if record:
        st.session_state['current_file'] = {
            "id": record[0],
            "file_name": record[1],
            "pinecone_index": record[2],
            "content": record[3]
        }
        
        st.session_state['messages'] = []        
        st.success(f"Loaded file: {record[1]} for analysis.")

def get_full_faq_text(file_index,pc):
    file_index_instance = pc.Index(file_index)
    res = file_index_instance.query(vector=[0.0] * 768, top_k=1000, include_metadata=True)
    return "\n".join([match["metadata"].get("text", "") for match in res.get("matches", [])])

def modify_query_for_web(query: str, context: str) -> str:
    mod_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    mod_prompt = f"""
    Given the FAQ Context: {context}
    And the user query: {query}
    Extract only the essential part of the query necessary for a web search.
    Provide a concise modified query focusing on key terms.
    """
    mod_response = mod_model.generate_content(mod_prompt).text
    return mod_response.strip()

def get_web_answer(query: str) -> str:
    prompt = f"Search the web and answer the following question: {query}"
    serp_agent = Agent(
        model=Gemini(model="gemini-2.0-flash", api_key=google_api),
        tools=[DuckDuckGo(), SerpApiTools(api_key=sa), WebsiteTools()],
        instructions=["Search the web for the most relevant and accurate information to answer the question briefly. Do not ask follow-up questions."]
    )
    try:
        response = serp_agent.run(prompt)
        web_answer = response.get_content_as_string()        
        return web_answer
    except Exception as e:
        return ""

def gen_summary(text: str) -> str:
    mo = genai.GenerativeModel(model_name="gemini-2.0-flash", 
                                system_instruction="You are a detailed report generator from the FAQ context. Generate a very detailed analysis report on every detail of the given FAQ context.")
    resp = mo.generate_content(f"Provide a very detailed analysis report regarding everything about the details and information covered based on the given FAQ document context\nFAQ Context: {text}").text
    return resp

def is_unsatisfactory(web_answer: str) -> bool:
    check_model = genai.GenerativeModel(model_name="gemini-2.0-flash", 
        system_instruction="""You are a sentence classifier. Your task is to analyze each provided sentence and determine whether it is "satisfactory" or "not satisfactory" based on the following criteria:
Satisfactory: The sentence conveys a positive meaning, includes clear and sufficient information, and directly provides the answer or solution.
Not Satisfactory: The sentence either lacks enough information, explicitly states that it does not have the answer (e.g., "I don't have the answer"), or fails to address the query effectively."""
    )
    few_shot_prompt = f"""
For each sentence, provide your classification along with a brief explanation of your reasoning.

Example 1:
Input Sentence: "The capital of France is Paris."
Output: Satisfactory – because the sentence gives a clear and correct answer along with necessary information.

Example 2:
Input Sentence: "I don't have the answer."
Output: Not Satisfactory – because it explicitly states a lack of an answer and provides no additional information.

Example 3:
Input Sentence: "I was unable to find the information regarding the data you requested."
Output: Not Satisfactory – because it explicitly states a lack of an answer and provides no additional information.

Example 4:
Input Sentence: "The provided FAQ document does not contain information about who founded ShopTalk. Since I was unable to find the answer using web search, I cannot answer who is the founder of shoptalk."
Output: Not Satisfactory – because it explicitly states a lack of an answer and provides no additional information.

Now, please classify the following sentences:
Sentences: "{web_answer}"
"""
    result = check_model.generate_content(few_shot_prompt).text.lower().strip()
    return "unsatisfactory" in result

def handle_query(query, history, faq_context, vector_store):
    summary = gen_summary(faq_context)  

    classification_prompt = f"""
    You are an FAQ chatbot. Analyze the provided FAQ context and the user query.
- If the query is not related to the topic of the FAQ context and also the history of the conversation, at all (i.e., less than 10% related), respond with "unrelated".
- If the query is related to the FAQ context and also the history of the conversation, but the answer is not available in the FAQ context, then label it as "needs web search". If after attempting web search the answer is still not found, label it as "keep in db".        
- If the query is answerable directly from the FAQ context, provide the answer using the context.
- If it's a follow-up question (e.g., "Tell me more", "Can you explain further", etc.), label it as "follow up".
    FAQ Context: {faq_context}
    User Query: {query}
    History : {history}
    """
    model = genai.GenerativeModel(model_name='gemini-2.0-flash')
    classification_response = model.generate_content(classification_prompt).text.lower()
    
    if "unrelated" in classification_response:
        alt_response = model.generate_content(f"""
Prompt:
You are a query classifier. Your task is to determine whether a given query is related to the provided reference content. The reference content can be any document—this may include FAQs, articles, bullet points, or any other format—and it can cover any topic.

Instructions:

Reference Content:
You will receive a piece of content. It could be in any format (plain text, JSON, bullet points, etc.) and about any topic or information.

Classification Task:
When you receive a query, use the following rules to classify it:

YES: If the query is related to the content provided.
Example: If the reference is about a workplace collaboration tool and the query asks, "Who is the founder of [tool]?" or "What is meant by the [particular feature] of the [tool] ?" it should be classified as YES, even if the founder's name isn’t mentioned.
NO: If the query is unrelated to the provided content.
Example: If the reference is about a workplace tool and the query asks, "What are Elon Musk's latest projects?" it should be classified as NO.
Output Format:
For each query, respond with either YES or NO (without any additional commentary).

Example Scenario:

Reference Content (any format):
"This document describes a new collaboration tool that helps teams manage projects, share files, and communicate effectively."

Query: "How can teams share files using this tool?"
Output: YES

Query : "What are similar products available in the market?"
Output: YES
                                              
Query: "What is the history of space exploration?"
Output: NO

Now, using these guidelines, classify the following query:
                                                       
{faq_context}
Query: {query}
Output (YES or NO):
""").text
        if "yes" in alt_response.lower():
            vector_store.add_texts([query], namespace="New Queries")
            return "This answer will be provided in the future."
        else:
            return "It is not related to the document."
    elif "needs web search" in classification_response:
        modified_query = modify_query_for_web(query, summary)
        web_answer = get_web_answer(modified_query)
        combined_prompt = f"""
        Answer the following user query using both the FAQ context and the web information.
        If the web information is not relevant or is unable to provide an answer, then perform your own web search and use your own knowledge to answer the User Query based on the provided FAQ Context.
        Consider the Chat history for conversation context.
        User Query: {query}
        FAQ Context: {summary}
        Web Information: {web_answer}
        Chat History: {history}

        Note : In the response, do not provide like, "As per the FAQ context, the answer is" or "Based on the provided FAQ Context". Instead, provide the answer directly.
        Note: In the response, do not provide anything like 'I found this information on the web' or 'I searched the web for you' or 'Based on the provided FAQ Context'. Instead, provide the answer directly.
        Note : The answer should always be on support of the FAQ context.
        """
        final_response = model.generate_content(combined_prompt).text
        if not final_response.strip() or is_unsatisfactory(final_response):
            vector_store.add_texts([query], namespace="New Queries")
            return "This answer will be provided in the future."
        else:
            vector_store.add_texts([f"Query: {query}\nResponse: {final_response}"], namespace="Web Queries")
            return final_response
    elif "follow up" in classification_response:
        follow_up_prompt = f"""
        Answer the user query based on the provided FAQ context and the chat history.
        User Query: {query}
        FAQ Context Summary: {summary}
        Chat History: {history}
        """
        return model.generate_content(follow_up_prompt).text
    else:
        prompt = f"User Query: {query}\nFAQ Context Summary: {summary}\nChat History: {history}\nProvide a direct answer."
        return model.generate_content(prompt).text

def store_conversation(user_id, file_id, role, content):
    c.execute("INSERT INTO conversations (user_id, file_id, role, content) VALUES (?, ?, ?, ?)",
              (user_id, file_id, role, content))
    conn.commit()

def main_app():
    st.set_page_config(page_title="FAQ Chatbot AI", layout="wide", page_icon="💬")
    st.title("FAQ Chatbot AI")
    
    pc_api = st.sidebar.text_input("Enter your Pinecone API Key : ")

    if pc_api:
        pc = create_pc(pc_api)

        st.sidebar.header("File Management")
        
        try:
            with st.sidebar.expander("Upload File"):
                handle_file_upload(pc)
        except Exception as e:
            st.error(f"Error: {e}")        
    
    else:
        st.sidebar.warning("Please enter a valid Pinecone API key")
    
    st.sidebar.header("Your Uploaded Files")
    c.execute("SELECT id, file_name FROM files WHERE user_id=?", (st.session_state['user_id'],))
    files = c.fetchall()
    for file_rec in files:
        file_id, file_name = file_rec
        if st.sidebar.button(f"Load: {file_name}", key=file_id):
            if pc_api:                
                load_file_for_analysis(file_id)
            else:
                st.sidebar.warning("Please enter a valid Pinecone API key, to load your files")
    
    tab1, tab2 = st.tabs(["Chatbot", "Uploaded Files"])
        
    with tab2:
        st.subheader("Uploaded Files")
        c.execute("SELECT id, file_name, file_content FROM files WHERE user_id=?", (st.session_state['user_id'],))
        files = c.fetchall()
        if files:
            for file_rec in files:
                file_id, file_name, file_content = file_rec
                with st.expander(f"{file_name} (File ID: {file_id})"):
                    st.text_area("File Content", file_content, height=300, key=f"{file_id}-file")
        else:
            st.write("No files uploaded yet.")
    
    with tab1:
        if st.session_state['current_file'] is None:
            st.warning("Please load a file from the sidebar to start analysis.")
        else:            
            try:
                vector_store = PineconeVectorStore(index=pc.Index(st.session_state['current_file']['pinecone_index']), embedding=embeddings)
                st.subheader(f"Chat with FAQ: {st.session_state['current_file']['file_name']}")
            
            
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                
                ori_user_query = st.chat_input("Ask your question here")
                faq_context = get_full_faq_text(st.session_state['current_file']['pinecone_index'],pc)
                history_str = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in st.session_state.messages])

                if ori_user_query:
                    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
                    user_query = model.generate_content(
                        f"""
                        Rewrite the given query to remove pronouns and clarify shortcuts.
                        Consider the following common shortcuts: "shld = should", "wt = what", "abt = about", "wdym = what do you mean", "exp = explain", "plz = please", "u = you", "r = are", "w = with", "w/o = without", "wrt = with respect to", "wrt = with regard to", "wrt = with reference to".
                        Example 1:
                        Input: What is shoptalk
                        Output: Who are shoptalk competitors?
                        Example 2:
                        Input: Explain the features of shoptalk
                        Output: Explain the features of shoptalk in detail
                        Example 3:
                        Input: What is the pricing of shoptalk
                        Output: Explain about the pricing of shoptalk in detail

                        User Query: {ori_user_query}
                        Chat History: {st.session_state.messages}
                        Output: Provide only the modified query as a single sentence."""
                    ).text.strip()
                    with st.chat_message("user"):
                        st.markdown(ori_user_query)
                    st.session_state.messages.append({"role": "user", "content": ori_user_query})
                                
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            bot_response = handle_query(user_query, history_str, faq_context, vector_store)
                            st.markdown(bot_response)
                            st.session_state.messages.append({"role": "assistant", "content": bot_response})
                            
                            store_conversation(st.session_state['user_id'], st.session_state['current_file']['id'], "assistant", bot_response)
                            store_conversation(st.session_state['user_id'], st.session_state['current_file']['id'], "user", ori_user_query)
            
            except Exception as e:
                st.warning("The Pinecone API key entered does not have the File uploaded, please use the respective API used when uploading the file")    

if not st.session_state['logged_in']:
    show_login_signup()

else:
    main_app()

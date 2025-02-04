import streamlit as st
import openai
import faiss
import numpy as np
import random
import pickle
import nbformat
from nbconvert import PythonExporter
import os


client = openai.OpenAI()


rag_application_path = "rag_application_fix.ipynb"

def load_rag_functions():
    with open(rag_application_path, "r", encoding="utf-8") as f:
        nb_content = nbformat.read(f, as_version=4)
    
    exporter = PythonExporter()
    source_code, _ = exporter.from_notebook_node(nb_content)
    
    exec(source_code, globals())

# RAG 관련 함수 로드
load_rag_functions()

# FAISS 벡터 데이터베이스 불러오기
faiss_index_path = "faiss_index.bin"
index = faiss.read_index(faiss_index_path)

# FAISS 데이터 로드 
faiss_data_path = "faiss_data.pkl"
try:
    with open(faiss_data_path, "rb") as f:
        faiss_data = pickle.load(f)  # 질문 데이터 로드
except FileNotFoundError:
    faiss_data = None

# FAISS에서 랜덤 문제 가져오는 함수
def get_random_question_from_faiss():
    if faiss_data is not None and len(faiss_data) > 0:
        retrieved_data = random.choice(faiss_data)
        if isinstance(retrieved_data, dict) and "question" in retrieved_data:
            question = retrieved_data["question"]
            options = retrieved_data.get("options", [])
            correct_answer = retrieved_data.get("answer", "")
            
            # 🔹 밑줄 처리 적용
            question = question.replace("effort", "<u>effort</u>")
            return question, options, correct_answer
    return None, None, None

# **Streamlit UI**
st.title("📘 RAG 기반 영어 학습 챗봇")

query_type = st.radio("검색 유형 선택", ["일반 질문", "랜덤 문제 풀기"])

if query_type == "일반 질문":
    query = st.text_input("질문을 입력하세요:")
    if st.button("응답 생성"):
        if query:
            with st.spinner("AI가 답변을 생성 중입니다..."):
                answer = generate_response(query)
            st.subheader("GPT-3.5의 답변")
            st.markdown(answer, unsafe_allow_html=True)

elif query_type == "랜덤 문제 풀기":
    
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
        st.session_state.current_options = []
        st.session_state.correct_answer = None
        st.session_state.answered = False

    if st.button("랜덤 문제 출제"):
        result = get_random_question_from_faiss()
        if result and all(result):
            st.session_state.current_question, st.session_state.current_options, st.session_state.correct_answer = result
            st.session_state.answered = False
    
    if st.session_state.current_question:
        st.subheader("📖 랜덤 문제")
        st.markdown(st.session_state.current_question, unsafe_allow_html=True)

        selected_option = st.radio("정답을 선택하세요:", st.session_state.current_options, index=None)

        if st.button("정답 확인"):
            if selected_option is None:
                st.warning("⚠️ 정답을 선택해주세요!")
            else:
                st.session_state.answered = True
                if selected_option == st.session_state.correct_answer:
                    st.success("✅ 정답입니다!")
                else:
                    st.error(f"❌ 오답입니다! 정답은: {st.session_state.correct_answer}")

        if st.session_state.answered:
            if st.button("새로운 문제 출제"):
                st.session_state.current_question = None
                st.session_state.current_options = []
                st.session_state.correct_answer = None
                st.session_state.answered = False
                st.rerun()

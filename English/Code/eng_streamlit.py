import streamlit as st
import openai
import faiss
import numpy as np
import random
import pickle
import nbformat
from nbconvert import PythonExporter

# ✅ OpenAI API 설정
client = openai.OpenAI()

# ✅ `rag_application.ipynb` 실행해서 함수 가져오기
rag_application_path = "rag_application.ipynb"

def load_rag_functions():
    with open(rag_application_path, "r", encoding="utf-8") as f:
        nb_content = nbformat.read(f, as_version=4)
    
    exporter = PythonExporter()
    source_code, _ = exporter.from_notebook_node(nb_content)
    
    exec(source_code, globals())

# ✅ RAG 관련 함수 로드
load_rag_functions()

# ✅ FAISS 벡터 데이터베이스 불러오기
faiss_index_path = "faiss_index.bin"
index = faiss.read_index(faiss_index_path)

# ✅ FAISS 데이터 로드 (`faiss_data.pkl`이 존재하면 로드)
faiss_data_path = "faiss_data.pkl"

try:
    with open(faiss_data_path, "rb") as f:
        faiss_data = pickle.load(f)  # 질문 데이터 로드
    print(f"✅ FAISS 데이터 로드 완료: 저장된 데이터 개수 {len(faiss_data)}")
except FileNotFoundError:
    faiss_data = None  # 데이터 파일이 없을 경우 None으로 설정
    print("⚠️ FAISS 데이터 파일이 없습니다. `faiss_data.pkl` 없이 동작하도록 설정됩니다.")

# ✅ FAISS에서 랜덤 문제 가져오는 함수
def get_random_question_from_faiss():
    """
    FAISS 벡터 데이터베이스에서 랜덤 문제를 가져옴.
    문제, 선택지(4개), 정답을 반환.
    """

    if faiss_data is not None and len(faiss_data) > 0:
        # ✅ `faiss_data.pkl`이 존재하면 랜덤 문제 선택
        retrieved_data = random.choice(faiss_data)

        # ✅ 데이터가 dict 형태인지 확인
        if isinstance(retrieved_data, dict) and "question" in retrieved_data and "options" in retrieved_data and "answer" in retrieved_data:
            question = retrieved_data["question"]
            options = retrieved_data["options"]
            correct_answer = retrieved_data["answer"]
        else:
            st.error("⚠️ FAISS 데이터 형식이 올바르지 않습니다. `faiss_data.pkl`을 확인하세요.")
            return None, None, None

    else:
        if index.ntotal == 0:
            st.error("⚠️ FAISS 인덱스가 비어 있습니다. 데이터가 올바르게 저장되었는지 확인하세요.")
            return None, None, None

        # ✅ FAISS 인덱스에서 랜덤 벡터 선택하여 검색
        random_idx = random.randint(0, index.ntotal - 1)
        query_vector = np.random.rand(1, index.d).astype(np.float32)  # 올바른 데이터 형태 변환

        # ✅ 가장 가까운 질문 벡터 검색
        distances, indices = index.search(query_vector, k=1)

        if indices[0][0] < 0:
            st.error("⚠️ FAISS 검색 결과가 없습니다. 데이터가 올바르게 저장되었는지 확인하세요.")
            return None, None, None

        # ✅ FAISS 검색 결과에서 올바른 질문 데이터를 가져옴
        question = f"질문 {indices[0][0] + 1}"  # FAISS에서 원본 데이터를 저장하지 않았다면 인덱스 번호만 표시
        options = ["A", "B", "C", "D"]
        correct_answer = random.choice(options)

    # 🔍 디버깅 출력
    print(f"🔍 검색된 문제: {question}, 선택지: {options}, 정답: {correct_answer}")

    return question, options, correct_answer


# 🎯 **Streamlit UI**
st.title("📘 RAG 기반 영어 학습 챗봇")

query_type = st.radio("검색 유형 선택", ["일반 질문", "랜덤 문제 풀기"])

# ✅ 일반 질문 처리
if query_type == "일반 질문":
    query = st.text_input("질문을 입력하세요:")
    if st.button("응답 생성"):
        if query:
            with st.spinner("AI가 답변을 생성 중입니다..."):
                answer = generate_response(query)  # ✅ `rag_application.ipynb`에서 불러옴
            st.subheader("GPT-3.5의 답변")
            st.write(answer)

# ✅ 랜덤 문제 풀기
elif query_type == "랜덤 문제 풀기":
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
        st.session_state.current_options = []
        st.session_state.correct_answer = None
        st.session_state.answered = False

    if st.button("랜덤 문제 출제"):
        result = get_random_question_from_faiss()

        # 🔍 디버깅용 출력
        print("🔍 get_random_question_from_faiss() 반환값:", result)
        st.write(f"🔍 디버깅: get_random_question_from_faiss() 반환값 = {result}")

        # 반환값 검증
        if result is None or not isinstance(result, tuple) or len(result) != 3:
            st.error("⚠️ FAISS 검색이 실패했습니다. 벡터 데이터가 올바르게 저장되었는지 확인하세요.")
            st.session_state.current_question = None
            st.session_state.current_options = []
            st.session_state.correct_answer = None
            st.stop()

        question, options, correct_answer = result

        if question and isinstance(question, str) and isinstance(options, list) and isinstance(correct_answer, str):
            if len(options) == 4:
                st.session_state.current_question = question
                st.session_state.current_options = options
                st.session_state.correct_answer = correct_answer
                st.session_state.answered = False
            else:
                st.error("⚠️ 문제의 선택지가 올바르지 않습니다. 데이터 형식을 확인하세요.")
                st.stop()
        else:
            st.error("⚠️ 반환된 문제 형식이 잘못되었습니다. RAG 데이터 처리를 확인하세요.")
            st.stop()

    if st.session_state.current_question:
        st.subheader("📖 랜덤 문제")
        st.write(st.session_state.current_question)

        # 선택지 표시
        selected_option = st.radio("정답을 선택하세요:", st.session_state.current_options, index=None, key="selected_option")

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
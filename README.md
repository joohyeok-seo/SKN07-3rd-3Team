# SKN07-3rd-3Team

# 📘 RAG 기반 영어 학습 챗봇
 
## 👥 팀 소개
| <img src="https://github.com/user-attachments/assets/b5a17c3c-8415-409b-ae90-0a931e677fc3" width="250" height="250"/> | <img src="https://github.com/user-attachments/assets/005f1a53-0700-420e-8c62-ae1555dd538b" width="250" height="260"/> | <img src="https://github.com/user-attachments/assets/3009a31a-d5ab-469a-bf39-39e6c7779efe" width="250" height="250"/>  | 
|:----------:|:----------:|:----------:|
| 영어 선생님 | 국어 선생님 | 사회 선생님 | 
| 서주혁 | 대성원 | 윤정연 | 

<br>

---

## 📖 프로젝트 개요
- 프로젝트 명:  RAG 기반 영어 학습 챗봇
- 프로젝트 소개: AI 고졸 검정고시 학습 튜터는 검정고시를 준비하는 학습자를 위한 서비스입니다. 2018년부터 2024년까지 7년치의 고졸 검정고시 기출 문제를 기반으로 하며 사용자가 질문을 입력하면 AI가 정답과 해설을 제공합니다.
  이 프로젝트는 **Retrieval-Augmented Generation (RAG)** 기술을 활용하여 영어 학습을 도와주는 **Streamlit 기반 챗봇**입니다. 사용자는 일반 질문을 입력하거나 FAISS 데이터베이스에서 랜덤으로 출제되는 문제를 풀 수 있습니다.
  
- 프로젝트 필요성(배경):
  - 기존의 검정고시 학습 자료는 개별적인 질문에 대한 즉각적인 응답을 받을 수 있는 시스템이 부족합니다.
  - 실제 기출된 문제만을 기반으로 설계되었으므로 학습자는 검정고시 대비에 있어 보다 체계적이고 효율적인 학습을 수행할 수 있습니다.
  - 경제적, 환경적 요인으로 인해 정규 교육을 받기 어려운 학습자들에게 공익적인 학습 기회를 제공하며, 교육 격차를 줄이는 데 기여할 수 있습니다.

- 프로젝트 목표:
  본 프로젝트는 AI 기술을 활용하여 학습자가 검정고시를 보다 효과적으로 준비할 수 있도록 돕는 것을 목표로 합니다.
  - Q&A 서비스 개발: 2018년부터 2024년까지의 고졸 검정고시 기출문제를 기반으로 CHAT GPT API를 활용하여 AI가 질문에 적절한 답변을 생성할 수 있도록 합니다.
  - 자연어 처리(NLP) 및 LLM(Chat GPT API) 활용: 텍스트 분석과 문서 이해 기술을 적용하여 질의응답 정확도를 향상합니다.
  - 사용자 친화적 인터페이스 제공: 학습자가 쉽게 접근하고 활용할 수 있는 직관적인 환경을 구축합니다.
  - 학습 효과 증대: AI를 활용한 실시간 응답 시스템으로 효율적인 학습을 지원합니다.
  - 맞춤형 학습 제공: 학습이 필요한 문제 유형을 분석하여 적절한 문제와 답, 해설을 제공하여 학습을 증진합니다.

---

## 🚀 주요 기능 소개

1. **일반 질문 응답**
   - 사용자가 입력한 질문에 대해 OpenAI GPT-3.5가 응답을 생성합니다.
2. **랜덤 문제 풀기**
   - FAISS 데이터베이스에서 영어 문제를 랜덤으로 불러와 사용자가 문제를 풀 수 있도록 합니다.
   - 선택지를 제공하며 정답을 확인할 수 있습니다.

---

## 📂 프로젝트 구조

<img src="https://github.com/user-attachments/assets/81339928-6c06-4bf1-8871-71630856ecac" alt="프로젝트 구조" width="800px">

---

## 🔧 기술 스택



---

## 📑 주요 프로시저

### ▶️ 1. 데이터 수집 _ 웹 크롤링 (한국교육과정평가원)
- 메인 페이지에서 각 연도, 학력분류, 차수, 과목별 코드번호 추출
```python
code_dict = {}
tmp = 1
while True:   # 모든 페이지 정보 가져오기
    url = f'https://www.kice.re.kr/boardCnts/list.do?type=default&page={tmp}&selLimitYearYn=Y&selStartYear=2018&C06=&boardID=1500211&C05=&C04=&C03=&searchType=S&C02=&C01='
    re = requests.get(url)
    soup = BeautifulSoup(re.text, 'html.parser')
    
    if soup.find('td').text.strip() == '등록된 게시물이 존재하지 않습니다.':
        break
        
    info = soup.find('tbody').find_all('tr')
    
    for i in info:
        code = i.find_all('td')[0].text
        year = i.find_all('td')[1].text
        edu = i.find_all('td')[2].text
        cnt = i.find_all('td')[3].text
        subject = i.find_all('td')[4].find('a')['title']
        # 원하는 자료 정보 선택
        if edu == '고졸학력' and subject == '영어':
            code_dict[code] = f'{year}_{edu}_{cnt}_{subject}'
    
    tmp += 1
```
![image](https://github.com/user-attachments/assets/be19cd7c-60ca-4ed9-a431-04b07a2ace09)
- 추출된 코드번호로 세부 페이지 접속 후 PDF 파일 저장
```python
for code in code_dict.keys():
    down_url = f'https://www.kice.re.kr/boardCnts/view.do?boardID=1500211&boardSeq={code}&lev=0&m=030305&searchType=S&statusYN=W&page=1&s=kice'
    down_re = requests.get(down_url)
    down_soup = BeautifulSoup(down_re.text)
    tmp_url = down_soup.find(class_='fieldBox').find('a')['href']
    pdf_url = 'https://www.kice.re.kr' + tmp_url
    file_path = f'./data/정답/{code_dict[code]}.pdf'
    
    response = requests.get(pdf_url)
    # 응답 상태 코드가 200(성공)인 경우에만 파일 저장
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)  # PDF 파일 저장
```
### ▶️ 2. 데이터 추출
### 2-1) 고등 영어 문제 추출

- PDF에서 왼쪽/오른쪽 문항을 올바른 순서로 정리하여 추출
```python
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        combined_text_list = []

        for page in pdf.pages:
            width, height = page.width, page.height

            # 왼쪽 문항 (페이지별 왼쪽 먼저)
            left_bbox = (0, 0, width / 2, height)
            left_crop = page.within_bbox(left_bbox)
            left_text = left_crop.extract_text()
            if left_text:
                combined_text_list.append(clean_text(left_text))  # 정리 후 추가

            # 오른쪽 문항 (페이지별 오른쪽 나중)
            right_bbox = (width / 2, 0, width, height)
            right_crop = page.within_bbox(right_bbox)
            right_text = right_crop.extract_text()
            if right_text:
                combined_text_list.append(clean_text(right_text))
```

- OCR 이미지 PDF 처리 (페이지별 왼쪽 → 오른쪽)
```python
def extract_text_from_image_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    combined_text_list = []

    for img in images:
        width, height = img.size

        # 왼쪽 문항 OCR
        left_crop = img.crop((0, 0, width // 2, height))
        left_text = pytesseract.image_to_string(left_crop, lang="eng+kor", config="--psm 6")
        combined_text_list.append(clean_text(left_text))

        # 오른쪽 문항 OCR
        right_crop = img.crop((width // 2, 0, width, height))
        right_text = pytesseract.image_to_string(right_crop, lang="eng+kor", config="--psm 6")
        combined_text_list.append(clean_text(right_text))

    return "\n".join(combined_text_list)
```

### 2-2) 고등 영어 정답 추출
- PDF에서 영어 정답 추출 (OCR 포함)
```python
def extract_english_answers_from_pdf(pdf_path):
    answers = {}

    # 1️⃣ PDF에서 직접 텍스트 추출
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    # 2️⃣ OCR 적용 (텍스트가 비어 있으면 OCR 사용)
    if not text.strip():
        text = extract_text_from_image_pdf(pdf_path)

    # 3️⃣ OCR로 추출한 원본 텍스트 출력 (디버깅 목적)
    print("\n📝 OCR EXTRACTED TEXT FROM PDF:", pdf_path)
    print(text[:1000])  # 처음 1000자만 출력

    # 4️⃣ "영어 정답표" 또는 "3교시 영어" 포함된 부분 찾기
    match = re.search(r"(?:영어 정답표|3교시 영어|영어)([\s\S]+?)(?=\n\w+ 정답표|\Z)", text)
    if match:
        english_answers_section = match.group(1).strip()
    else:
        print(f"⚠ 영어 정답을 찾을 수 없음: {pdf_path}")
        return None

    # 5️⃣ 정답 패턴 추출 (디버깅용 출력 추가)
    extracted_text = convert_korean_numbers(english_answers_section)
    print("\n🔍 EXTRACTED ENGLISH ANSWERS SECTION:")
    print(extracted_text[:500])  # 처음 500자만 출력

    # 6️⃣ 문항번호 & 정답 추출
    answer_pattern = re.findall(r"(\d+)\s+([①②③④1-4])", extracted_text)

    # 🔥 디버깅: 추출된 정답 출력
    print("\n🔎 Extracted Answers Dictionary:", answer_pattern)

    for q_num, ans in answer_pattern:
        answers[q_num] = ans

    return answers
```

### 2-3) 고등 영어 문제, 정답 json파일 합치기
- 파일명 정리 (정답 파일명과 문제 파일명 일치하도록 변환)
```python
def clean_filename(filename):
    return filename.replace("_고등_정답.pdf", "_고등_영어.pdf")  # 정답 파일명을 문제 파일명과 맞춤
```

- 변환된 정답 데이터 키 값 수정
```python
answers_data_fixed = {clean_filename(k): v for k, v in answers_data.items()}\
```

- 문제와 정답 매칭
```python
merged_data = {}

for file_name, question_content in questions_data.items():
    matched_file = clean_filename(file_name)
    if matched_file in answers_data_fixed:  # 정답이 있는 경우만 추가
        merged_data[file_name] = {
            "questions": question_content,
            "answers": answers_data_fixed[matched_file]
        }
    else:
        print(f"⚠ 정답이 없는 문제 파일: {file_name}")
```

### ▶️ 3. 임베딩
- OpenAI의 "text-embedding-ada-002" 모델를 사용해 질문을 벡터로 변환
```
# OpenAI 임베딩 함수 (최신 API 적용)
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding  # 최신 API 방식 적용

# 모든 질문을 임베딩 변환
question_embeddings = [get_embedding(q) for q in questions]
```

- FAISS 데이터베이스 생성
```
embedding_dim = 1536  # 벡터 차원 설정
index = faiss.IndexFlatL2(embedding_dim)  # FAISS 인덱스 생성
question_vectors = np.array(question_embeddings).astype("float32")  # 임베딩 데이터를 numpy 배열로 변환
index.add(question_vectors)  # FAISS 데이터베이스에 추가
```

- FAISS를 이용한 유사 질문 검색
```
def search_similar_questions(query, top_k=3):
    # 입력 질문을 벡터로 변환
    query_vector = np.array(get_embedding(query)).astype("float32").reshape(1, -1)

    # 가장 가까운 질문 검색
    distances, indices = index.search(query_vector, top_k)

    # 결과 출력
    print("\n[가장 유사한 질문들]")
    for i in range(top_k):
        idx = indices[0][i]
        print(f"{i+1}. {questions[idx]} (거리: {distances[0][i]:.4f})")
```

- FAISS 인덱스 저장
```
faiss.write_index(index, "faiss_index.bin")
```
---

## 🎬수행결과(테스트/시연 페이지)

---
 
## 💭한 줄 회고

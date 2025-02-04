# SKN07-3rd-3Team

# AI 고졸 검정고시 학습 튜터
 
# 1. 팀 소개
| <img src="https://github.com/user-attachments/assets/b5a17c3c-8415-409b-ae90-0a931e677fc3" width="250" height="250"/> | <img src="https://github.com/user-attachments/assets/005f1a53-0700-420e-8c62-ae1555dd538b" width="250" height="260"/> | <img src="https://github.com/user-attachments/assets/3009a31a-d5ab-469a-bf39-39e6c7779efe" width="250" height="250"/>  | 
|:----------:|:----------:|:----------:|
| 영어 선생님 | 국어 선생님 | 사회 선생님 | 
| 서주혁 | 대성원 | 윤정연 | 


# 2. 프로젝트 개요
- 프로젝트 명: AI 고졸 검정고시 학습 튜터
- 프로젝트 소개: AI 고졸 검정고시 학습 튜터는 검정고시를 준비하는 학습자를 위한 서비스입니다. 2018년부터 2024년까지 7년치의 고졸 검정고시 기출 문제를 기반으로 하며 사용자가 질문을 입력하면 AI가 정답과 해설을 제공합니다. 
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

 
# 3. 기술 스택

## 주요 프로시저

1) 데이터 수집 _ 웹 크롤링 (한국교육과정평가원)
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

## 수행결과(테스트/시연 페이지)
 
## 한 줄 회고

## 1

전처리 모듈은 의료 데이터를 고정 길이의 숫자 벡터로 변환하고, 향후 OpenFHE CKKS 암호화 연산에 사용할 데이터를 생성합니다.

- **데이터셋:** UCI Heart Disease (Cleveland)
- **데이터 규모:** 303개 기록
- **입력 특징:** 13개
- **벡터 차원:** 32차원
- **검색 방식:** Cosine Similarity 기반 Top-5
- **개발 언어:** Python
- **주요 라이브러리:** Pandas, NumPy, Scikit-learn, ucimlrepo

---

## 2. 실행 방법

### ① 프로젝트 폴더로 이동

```bash
cd ~/CyperCare/preprocessing
```

### ② 가상환경 설정 및 라이브러리 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

가상환경을 이미 생성했다면 활성화만 하면 됩니다

```bash
source .venv/bin/activate
```

### ③ UCI 데이터 다운로드 및 전처리

```bash
python preprocess.py --pad-to 32 --query-count 20
```

실행하면 `fetch_ucirepo(id=45)`를 이용해 실제 UCI Heart Disease 데이터를 불러옵니다.

macOS에서 SSL 인증서 오류가 발생한다면 다음 명령어를 먼저 실행합니다.

```bash
export SSL_CERT_FILE="$(python -m certifi)"
```

### ④ 결과 검증

```bash
python check_vectors.py
```

등록 벡터와 Query의 크기, L2 정규화 여부를 확인하고 첫 번째 Query의 Top-5 검색 결과를 출력합니다.

### ⑤ 새로운 Query 전처리

```bash
python transform_query.py --csv sample_queries/heart_query_example.csv
```

저장된 전처리기를 재사용해 새로운 환자 입력에도 동일한 변환 과정을 적용합니다.

---

## 3. 데이터 전처리 과정

### Step 1. 데이터 로드 및 특징 선택

UCI Heart Disease 데이터셋의 13개 의료 특징을 사용합니다.

질환 여부를 나타내는 레이블 `num`은 검색 특징에서 제외합니다.

**수치형 특징**

`age`, `trestbps`, `chol`, `thalach`, `oldpeak`, `ca`

**범주형 특징**

`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `thal`

### Step 2. 데이터 분할

총 303개 기록을 다음과 같이 나눕니다.

- 등록 데이터베이스: 283개
- 검색용 Query: 20개

난수 시드 `42`를 사용하여 동일한 결과를 재현할 수 있도록 구성했습니다.

### Step 3. 결측치 처리

`?` 또는 빈 값을 결측치로 처리합니다.

- 수치형 데이터: 등록 데이터베이스의 중앙값(Median)으로 대체
- 범주형 데이터: 결측치를 `__MISSING__` 범주로 표시

### Step 4. 수치형 데이터 스케일링

`StandardScaler`를 적용해 수치형 특징의 평균을 0, 표준편차를 1 기준으로 표준화합니다.

나이, 혈압, 콜레스테롤처럼 단위와 범위가 다른 값이 검색 결과에 과도한 영향을 미치는 문제를 줄이기 위한 과정입니다.

### Step 5. One-Hot Encoding

범주형 의료 특징에 `OneHotEncoder`를 적용하여 숫자 벡터로 변환합니다.

현재 실제 데이터에서는 인코딩 이후 **26차원 벡터**가 생성되었습니다.

전처리 규칙은 등록 데이터베이스에서 학습하고, Query에도 동일하게 적용합니다.

### Step 6. Zero Padding

인코딩된 26차원 벡터의 끝에 0을 6개 추가해 32차원으로 변환합니다.

```text
Encoded Vector: 26 dimensions
        ↓
Zero Padding: +6 dimensions
        ↓
Final Vector: 32 dimensions
```

이 과정은 암호문 Packing 및 차원별 연산 실험을 위한 것으로, 새로운 의료 정보를 추가하지 않습니다.

### Step 7. L2 Normalization

각 벡터의 L2 Norm이 1이 되도록 정규화합니다.

정규화된 두 벡터의 내적은 코사인 유사도와 동일합니다.

\[
\operatorname{CosSim}(q,r)=
\frac{q\cdot r}{\|q\|\|r\|}
\]

벡터의 길이가 1이므로 다음과 같이 계산할 수 있습니다.

\[
\operatorname{CosSim}(q,r)=q\cdot r
\]

따라서 향후 CKKS를 이용할 때 암호화된 벡터의 내적 연산으로 유사도 점수를 계산할 수 있습니다.

### Step 8. Plaintext Top-K 검색

Query 벡터와 등록 데이터베이스의 모든 벡터에 대해 코사인 유사도를 계산합니다.

유사도가 높은 순서대로 정렬하여 상위 5개의 기록을 반환합니다.

이 결과는 향후 OpenFHE로 계산한 암호화 검색 결과와 비교하는 기준값으로 사용됩니다.

---

## 4. 실제 실행 결과

macOS 환경에서 실제 UCI Heart Disease 데이터를 다운로드하여 전처리 및 검증을 수행했습니다

### 데이터 전처리 결과

```text
Downloading UCI Heart Disease (ID 45) ...
Loaded Heart Disease: 303 records
Registered: (283, 32); queries: (20, 32)
Encoded dimension: 26; final dimension: 32
Saved to: /Users/shu/CyperCare/preprocessing/data/processed/heart
```

| 항목 | 결과 |
|---|---|
| 전체 기록 | 303개 |
| 등록 데이터 | 283개 |
| Query | 20개 |
| 인코딩 후 차원 | 26 |
| 최종 벡터 차원 | 32 |
| 검색 결과 | Top-5 |

### 첫 번째 Query의 검색 결과

```text
Dataset: UCI Heart Disease
Database: 283 records; queries: 20
Vector dimension: 32

Query #1 | Top-5:
  1. H00113  cosine=0.927677
  2. H00173  cosine=0.846183
  3. H00145  cosine=0.712733
  4. H00091  cosine=0.707680
  5. H00269  cosine=0.697817

PASSED: vector shapes and L2 norms verified.
```

첫 번째 Query와 가장 유사한 기록은 `H00113`이며, 코사인 유사도는 **0.927677**로 계산되었습니다.
검증 결과 데이터베이스와 Query의 벡터 차원이 일치하고, 모든 벡터가 L2 정규화 조건을 만족했습니다.

### 현재 구현 상태

- [x] UCI Heart Disease 실제 데이터 다운로드
- [x] 결측치 처리
- [x] 수치형 데이터 스케일링
- [x] 범주형 데이터 One-Hot Encoding
- [x] 32차원 벡터 생성
- [x] L2 Normalization
- [x] Plaintext Cosine Similarity Top-5 검색
- [x] 벡터 차원 및 정규화 검증
- [ ] OpenFHE CKKS 암호화 검색
- [ ] Blind-Match Input Expansion 및 HE-C³ 적용
- [ ] 암호화 검색 성능 평가

---

## 5. 생성 파일

전처리 결과는 `data/processed/heart/`에 저장됩니다.

| 파일 | 설명 |
|---|---|
| `database.npy` | 등록 데이터 벡터 (283 × 32) |
| `queries.npy` | 검색 Query 벡터 (20 × 32) |
| `database_vectors.csv` | 등록 벡터 CSV, 헤더 없음 |
| `query_vectors.csv` | Query 벡터 CSV, 헤더 없음 |
| `database_ids.csv` | 등록 기록 ID |
| `query_ids.csv` | Query ID |
| `plaintext_topk.csv` | Top-5 검색 결과 및 유사도 |
| `preprocessor.joblib` | 학습된 전처리기 |
| `metadata.json` | 차원 및 전처리 설정 정보 |

생성 파일은 암호화 이전의 평문 데이터입니다. 외부 검색 서버에는 암호화한 벡터만 전달해야 합니다.

---
## References

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [Blind-Match Paper](https://arxiv.org/abs/2408.06167)
- [Blind-Match GitHub](https://github.com/hm-choi/blind-match)
- [OpenFHE](https://github.com/openfheorg/openfhe-development)

---
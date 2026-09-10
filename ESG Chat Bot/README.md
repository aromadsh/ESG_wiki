# ESG 챗봇

Python 실행 및 라이브러리 설치는 로컬 Anaconda `esg` 환경을 사용합니다.

## 파일 역할

| 파일·폴더 | 역할 |
|---|---|
| `preprocessing.py` | CSV 통합·정제, 문자 사전과 학습 배열 생성, 텍스트 분석 모듈 |
| `seq2seq.ipynb` | 로컬 실행: 전처리 모듈 호출 → 분석 그래프 → 학습·저장 → 질의응답 |
| `test.ipynb` | Colab용 통합 노트북. 전처리 코드 내장, 별도 Python 파일 불필요 |
| `data_in/` | 전처리 배열·설정·분석 결과 |
| `data_out/` | 저장 모델·설정·학습 기록 |

## 로컬 실행

1. `seq2seq.ipynb`를 열고 `esg` 커널을 선택합니다.
2. `MODE = "train"`, `PREPARED_DIR = None`으로 전체 실행합니다.
3. `preprocessing.py`가 프로젝트 루트의 `ChatbotData.csv`와 `ESG_QnA_dataset_10000.csv`를 처리합니다.
4. 데이터 분석 그래프 및 학습 loss/accuracy 그래프를 노트북에서 확인합니다.
5. 마지막 질문 셀을 수정하거나 `INTERACTIVE = True`로 대화합니다. `/quit`으로 종료합니다.
6. 다음에는 `MODE = "chat"`으로 실행하면 최신 저장 모델을 불러옵니다.

전처리 설정은 `DATA_PATHS`, `MAX_LENGTH`, `MAX_SAMPLES`입니다. 기존 전처리를 재사용하려면 `PREPARED_DIR`에 해당 결과 폴더를 지정하세요. 데이터가 변경되면 `None`으로 전처리를 다시 실행해야 합니다.

빠른 점검은 `MAX_SAMPLES = 128`, `EPOCHS = 1`, 본 학습은 `MAX_SAMPLES = None`, `EPOCHS = 20`으로 설정합니다. 기본 길이 256에서는 답변이 종료 토큰을 제외한 255자까지만 학습됩니다. 잘리는 문항 수가 표시되며, 길이를 늘리면 메모리와 학습 시간도 증가합니다.

## Colab 실행

1. Colab에서 `test.ipynb`를 열고 순서대로 실행합니다.
2. 학습 모드에서 요청 시 두 CSV를 업로드합니다.
3. 전처리·분석·학습·저장·질의응답을 진행합니다.
4. 마지막 셀에서 모델 ZIP을 다운로드해 보관합니다.
5. 새 런타임의 `MODE = "chat"`에서는 ZIP 안의 `best.weights.h5`와 `config.json`을 업로드합니다.

Colab에서는 런타임 라이브러리를 사용하며 자동 설치하지 않습니다. 런타임의 파일은 영구 보관되지 않습니다. 로컬 TensorFlow 2.16.2에서 소량 실행을 검증했으며 실제 Colab 환경 실행은 별도 확인이 필요합니다.

## 데이터 및 지표

빈 행과 동일 Q/A 중복을 제외합니다. 같은 입력 질문은 한 그룹으로 묶어 그룹 수 기준 약 9:1로 학습·검증을 분리합니다. 문자 사전은 학습 데이터에서만 만듭니다. 단어 빈도는 표면형 집계이며 형태소·의미 분석은 아닙니다.

`loss`, `accuracy`, `val_loss`, `val_accuracy`를 기록합니다. 정확도는 PAD를 제외하고 EOS를 포함한 문자 토큰 기준으로, 답변의 사실성을 의미하지 않습니다. 최적 모델은 `val_loss` 기준으로 저장합니다.

모델 복원에는 `best.weights.h5`와 `config.json`이 함께 필요합니다. 원본 CSV와 저장 모델은 정리 대상에서 제외했습니다.
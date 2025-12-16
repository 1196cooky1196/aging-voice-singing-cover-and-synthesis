# aging-voice-singing-cover-and-synthesis

### 🧠 Model Architecture
```mermaid
graph TD
    %% 스타일 정의
    classDef input fill:#e1bee7,stroke:#4a148c,stroke-width:2px;
    classDef gen fill:#bbdefb,stroke:#0d47a1,stroke-width:2px;
    classDef fake fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px;
    classDef frozen fill:#f5f5f5,stroke:#616161,stroke-width:2px,stroke-dasharray: 5 5;
    classDef loss fill:#ffcdd2,stroke:#c62828,stroke-width:2px;

    %% 1. 메인 흐름 (CycleGAN)
    subgraph Main_Flow ["Main Flow"]
        InputA("Input A (Youth Voice)"):::input
        G_AB["Generator A to B"]:::gen
        FakeB("Fake B (Aged Voice)"):::fake
        G_BA["Generator B to A"]:::gen
        RecA("Reconstructed A"):::fake
    end

    InputA --> G_AB --> FakeB --> G_BA --> RecA

    %% 2. 가이드 모듈 (Frozen & Student)
    subgraph Guidance ["Guidance Modules"]
        SE["Speaker Encoder (Frozen)"]:::frozen
        Teacher["Keras Teacher (Frozen)"]:::frozen
        AgeHead["Age Head (Student)"]:::gen
    end

    %% 3. 손실 함수 (Losses)
    subgraph Loss_Function ["Loss Functions"]
        L_Cyc["Cycle Loss"]:::loss
        L_ID["Speaker ID Loss"]:::loss
        L_KD["Age KD Loss"]:::loss
    end

    %% --- 연결선 ---
    %% Cycle Loss
    InputA -.-> L_Cyc
    RecA -.-> L_Cyc

    %% Speaker Identity
    InputA --> SE
    FakeB --> SE
    SE -.-> L_ID

    %% Knowledge Distillation
    FakeB --> Teacher
    FakeB --> AgeHead
    Teacher -.-> L_KD
    AgeHead -.-> L_KD

<br>

### 🔍 System Overview
본 프로젝트는 비병렬(Unpaired) 데이터 환경에서 목소리의 **화자 고유성(Identity)은 유지**하면서 **나이(Age) 특성만 변환**하는 것을 목표로 합니다. 이를 위해 CycleGAN 베이스에 두 가지 핵심 제약 조건을 추가했습니다.

1.  **CycleGAN Backbone:** * Cycle Consistency Loss를 통해 A(청년) → B(중장년) → A(청년) 복원 과정을 학습하며, 변환 과정에서 언어 정보가 유실되지 않도록 합니다.
2.  **Speaker Consistency (화자 일관성 유지):**
    * Pre-trained Speaker Encoder를 사용하여 변환 전후의 화자 임베딩 유사도(Cosine Similarity)를 계산합니다.
    * 이를 통해 목소리가 늙더라도 **"누구인지(Identity)"**는 변하지 않도록 제어합니다.
3.  **Age Knowledge Distillation (나이 특징 주입):**
    * 별도로 학습된 고성능 나이 예측 모델(Keras Teacher)의 지식을 경량화된 Age Head(Student)에 증류(Distillation)합니다.
    * Generator가 생성한 목소리가 목표 나이대(Target Age Group)의 특징을 정확히 갖도록 강제합니다.

---

### 📂 Dataset Construction
본 연구는 **YouTube**에서 추출한 고품질 보컬 데이터를 사용하여 구축되었습니다.

* **Data Source:** YouTube Music 
* **Composition:**
    * **Domain A (Source):** 20대 가수 30명 (남/녀 균형 구성)
    * **Domain B (Target):** 40~60대 중장년 가수 30명
* **Preprocessing:**
    * 모든 음원은 16kHz로 리샘플링되었습니다.
    * 무음 구간(Silence)을 제거한 후, 학습 효율을 위해 **5초(5.0s) 단위로 슬라이싱(Slicing)** 처리하였습니다.

> **⚠️ Copyright Notice** > 본 프로젝트에 사용된 데이터셋은 저작권 문제로 인해 본 리포지토리에 포함되지 않습니다.  
> The dataset is **NOT** uploaded due to copyright restrictions.

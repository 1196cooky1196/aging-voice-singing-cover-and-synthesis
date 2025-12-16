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



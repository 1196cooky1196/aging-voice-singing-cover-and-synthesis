# aging-voice-singing-cover-and-synthesis

### 🧠 Model Architecture (CycleGAN + KD + Speaker Consistency)
```mermaid
graph TD
    %% --- 스타일 정의 (가독성 높임) ---
    classDef input fill:#e1bee7,stroke:#4a148c,stroke-width:2px,color:#000;
    classDef gen fill:#bbdefb,stroke:#0d47a1,stroke-width:2px,color:#000;
    classDef fake fill:#c8e6c9,stroke:#1b5e20,stroke-width:2px,color:#000;
    classDef frozen fill:#f5f5f5,stroke:#616161,stroke-width:2px,stroke-dasharray: 5 5,color:#000;
    classDef loss fill:#ffcdd2,stroke:#c62828,stroke-width:2px,color:#000;

    %% --- 1. 메인 흐름 (CycleGAN) ---
    subgraph Main Flow
        InputA(🎤 Input A<br>Youth Voice):::input
        G_AB[Generator A→B]:::gen
        FakeB(⚡ Fake B<br>Converted Aged Voice):::fake
        G_BA[Generator B→A]:::gen
        RecA(🔄 Reconstructed A):::fake
    end

    InputA --> G_AB --> FakeB --> G_BA --> RecA

    %% --- 2. 제약 조건 (Losses & Guidance) ---
    subgraph Guidance Modules
        SE[Speaker Encoder<br>(Frozen)]:::frozen
        Teacher[Keras Teacher Model<br>(Frozen)]:::frozen
        AgeHead[Age Head<br>(Student Trainable)]:::gen
    end

    subgraph Losses
        Loss_Cycle(Cycle Consistency Loss):::loss
        Loss_ID(Speaker Identity Loss):::loss
        Loss_KD(Age Knowledge Distillation Loss):::loss
    end

    %% --- 연결선 (흐름 및 Loss 계산) ---
    %% Cycle Loss
    InputA -.-> Loss_Cycle
    RecA -.-> Loss_Cycle

    %% Speaker Identity Loss (화자 유지)
    InputA --> SE
    FakeB --> SE
    SE -.-> Loss_ID

    %% Knowledge Distillation (나이 변환)
    FakeB --> Teacher -.-> Loss_KD
    FakeB --> AgeHead -.-> Loss_KD


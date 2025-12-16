# aging-voice-singing-cover-and-synthesis

### 🧠 Model Architecture (CycleGAN + KD + Age Constraint)
```mermaid
graph TD
    %% 스타일 정의
    classDef input fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef gen fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef fake fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef frozen fill:#eeeeee,stroke:#616161,stroke-width:2px,stroke-dasharray: 5 5;
    classDef student fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef loss fill:#ffffff,stroke:#ff0000,stroke-width:2px,stroke-dasharray: 5 5;

    %% 1. 입력 데이터
    InputA(🎤 Input A<br>Youth Voice):::input
    InputB(🎤 Input B<br>Aged Voice):::input

    %% 2. CycleGAN Main Loop
    subgraph CycleGAN [CycleGAN Backbone]
        direction TB
        GA2B[Generator A→B]:::gen
        GB2A[Generator B→A]:::gen
        FakeB[⚡ Fake B<br>Converted Aged Voice]:::fake
        RecA[🔄 Reconstructed A]:::fake
    end

    %% Flow 연결
    InputA --> GA2B --> FakeB --> GB2A --> RecA
    
    %% 3. Speaker Consistency (화자 유지)
    subgraph Spk_Consist [Speaker Consistency]
        SE[Speaker Encoder<br>(Frozen)]:::frozen
        StyleA[Style Vector A]
        StyleFake[Style Vector Fake]
    end
    InputA --> SE --> StyleA
    FakeB --> SE --> StyleFake
    StyleA -.-> |"📉 Cosine Sim Loss<br>(Keep Identity)"| StyleFake
    class StyleA,StyleFake loss

    %% 4. Knowledge Distillation (나이 변환)
    subgraph KD_Module [Age Knowledge Distillation]
        direction TB
        AgeHead[Age Head<br>(Student)]:::student
        Teacher[Keras Teacher<br>(Pre-trained Age Classifier)]:::frozen
        Stats[Feature Stats<br>(111 dim)]
    end

    %% KD 연결
    FakeB --> AgeHead
    FakeB --> |Calc Stats| Stats --> Teacher
    AgeHead -.-> |"📉 Grouped CE Loss<br>(Force Target Age)"| AgeHead
    Teacher -.-> |"📉 KD Loss<br>(Mimic Teacher)"| AgeHead
    
    %% Cycle Loss
    RecA -.-> |"📉 Cycle Loss (L1)"| InputA
    class RecA loss

### 🔎 다이어그램 설명 
코드가 복잡해 보이지만, 핵심은 **3가지 Loss**가 서로 견제하며 학습하는 구조입니다. 면접 때 이 그림을 띄워놓고 이렇게 설명하면 됩니다.

1.  **CycleGAN Backbone (파란색):**
    * 기본적인 A → B → A 변환을 수행하며, `Cycle Loss`를 통해 원래 목소리로 복구 가능한지 확인합니다.
2.  **Speaker Consistency (회색/점선):**
    * `Speaker Encoder`는 학습되지 않는(Frozen) 상태로 둡니다.
    * 변환된 목소리(`Fake B`)가 원래 목소리(`Input A`)의 **화자 특성(Identity)을 잃어버리지 않았는지** 코사인 유사도로 감시합니다. (이게 있어서 목소리가 안 깨지는 겁니다.)
3.  **Knowledge Distillation (주황색):**
    * `Keras Teacher`는 이미 나이를 잘 맞추는 모델입니다.
    * 우리의 `Generator`가 만든 목소리를 Teacher에게 보여주고, **"이거 50대 목소리 맞아?"** 라고 검사받습니다(`KD Loss`).
    * 동시에 `Age Head`(Student)를 통해 강제로 나이 그룹(Grouped CE)을 맞추도록 학습합니다.

이 구조도는 `train.py`의 `CycleGAN++(patched)` 로직을 정확하게 시각화한 것입니다.

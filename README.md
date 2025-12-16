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


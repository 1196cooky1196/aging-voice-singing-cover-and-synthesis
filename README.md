# aging-voice-singing-cover-and-synthesis

### 🏗️ Model Architecture
```mermaid
graph LR
    %% 스타일 정의 (색상 및 디자인)
    classDef input fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef teacher fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef student fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef loss fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,stroke-dasharray: 5 5;

    %% 1. 입력 데이터
    Input(🎤 Input Audio<br>Original Singing Voice):::input

    %% 2. Teacher Model (고성능 CycleGAN)
    subgraph Teacher_Model [Teacher: Aging CycleGAN]
        direction TB
        T_Enc[Encoder] --> T_Res[ResNet Blocks x9]
        T_Res --> T_Dec[Decoder]
    end

    %% 3. Student Model (경량화 모델)
    subgraph Student_Model [Student: Lightweight Model]
        direction TB
        S_Enc[Encoder] --> S_Res[ResNet Blocks x3]
        S_Res --> S_Dec[Decoder]
    end

    %% 4. 흐름 연결
    Input --> T_Enc
    Input --> S_Enc

    %% ★핵심: 지식 증류 (KD) 연결★
    T_Res -.-> |"📉 KD Loss<br>(Feature Matching)"| S_Res
    class T_Res,S_Res loss

    %% 5. 최종 출력
    T_Dec --> Out_T(👴 High Quality<br>Aged Voice)
    S_Dec --> Out_S(⚡ Compressed<br>Aged Voice):::student

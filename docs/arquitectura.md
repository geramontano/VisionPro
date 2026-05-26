# System architecture

VisionPro is organized as a modular real time computer vision pipeline. The system combines facial emotion recognition, gaze estimation, distance approximation and behavioral logging.

```mermaid
flowchart TD
    A["Webcam input<br/>OpenCV VideoCapture"] --> B["Frame acquisition<br/>src/app.py"]
    B --> C["Face detection<br/>src/cara.py"]
    C --> D{"Face detected?"}

    D -- "No" --> D1["Show no face status<br/>Skip prediction"]
    D -- "Yes" --> E["Face crop<br/>Normalized facial region"]

    E --> F["Feature extraction<br/>src/rasgos.py"]
    F --> F1["Texture features<br/>HOG descriptors"]
    F --> F2["Texture features<br/>LBP descriptors"]
    F --> F3["Geometric features<br/>eyes mouth eyebrows proportions"]

    F1 --> G["Feature vector x_t"]
    F2 --> G
    F3 --> G

    G --> H["Emotion ensemble model<br/>modelo/emociones_modelo.joblib"]
    H --> H1["SVM RBF"]
    H --> H2["ExtraTrees"]
    H --> H3["RandomForest"]
    H --> H4["LogisticRegression"]
    H --> H5["MLPClassifier"]

    H1 --> I["Class probability fusion"]
    H2 --> I
    H3 --> I
    H4 --> I
    H5 --> I

    I --> J["Auxiliary facial rules<br/>src/reglas.py"]
    J --> K["Probability adjustment<br/>model plus rules"]
    K --> L["Temporal smoothing<br/>src/modelo.py"]
    L --> M["Emotion prediction<br/>neutral happy sad angry afraid surprised"]
    L --> M1["Live confidence score"]

    C --> N["Eye region extraction<br/>src/mirada.py"]
    N --> O["Pupil and eye feature estimation"]
    O --> P{"Gaze calibrated?"}
    P -- "No" --> P1["Use raw approximate gaze"]
    P -- "Yes" --> Q["Multipoint gaze regression<br/>calibracion_mirada.npz"]
    P1 --> R["Estimated gaze point"]
    Q --> R
    R --> S["Screen zone estimation<br/>top middle bottom<br/>left center right"]

    C --> T["Face size measurement"]
    T --> U{"Distance calibrated?"}
    U -- "No" --> U1["Distance unavailable or approximate"]
    U -- "Yes" --> V["Distance estimation<br/>d_t = d_0 h_0 / h_t"]
    U1 --> W["Camera distance status"]
    V --> W

    M --> X["Runtime display overlay"]
    M1 --> X
    R --> X
    W --> X

    M --> Y["CSV logger<br/>registros/mirada_emociones_v2.csv"]
    M1 --> Y
    R --> Y
    S --> Y
    W --> Y

    Y --> Z["Post processing<br/>src/analizar_mirada.py"]
    Z --> Z1["Emotion distribution by screen zone"]
    Z --> Z2["Average confidence by zone"]
    Z --> Z3["Attention and response summary"]

    AA["Public datasets<br/>JAFFE KDEF RAVDESS"] --> AB["Dataset preparation<br/>src/preparar_datasets.py"]
    AB --> AC["Prepared dataset<br/>datasets/preparado"]
    AC --> AD["Training script<br/>src/entrenar.py"]
    AD --> H
    AD --> AE["Validation report<br/>reportes/reporte_entrenamiento.txt"]

    classDef input fill:#0B3D5C,stroke:#7DD3FC,color:#F8FAFC,stroke-width:1.5px;
    classDef process fill:#312E81,stroke:#A5B4FC,color:#F8FAFC,stroke-width:1.5px;
    classDef model fill:#7C2D12,stroke:#FDBA74,color:#FFF7ED,stroke-width:1.5px;
    classDef output fill:#14532D,stroke:#86EFAC,color:#F0FDF4,stroke-width:1.5px;
    classDef decision fill:#713F12,stroke:#FDE68A,color:#FFFBEB,stroke-width:1.5px;
    classDef storage fill:#1F2937,stroke:#CBD5E1,color:#F8FAFC,stroke-width:1.5px;

    class A,AA input;
    class B,C,E,F,F1,F2,F3,J,K,L,N,O,Q,R,S,T,V,Z process;
    class H,H1,H2,H3,H4,H5,I,AD model;
    class X,Z1,Z2,Z3,M,M1,W output;
    class D,P,U decision;
    class Y,AB,AC,AE storage;
```

## Module summary

The architecture separates training and runtime execution.

During training, public datasets are prepared and transformed into face crops. Then the emotion ensemble is trained and saved as a Joblib model.

During runtime, the webcam stream is processed frame by frame. The system detects the face, extracts features, predicts emotion, estimates gaze and distance, and stores synchronized interaction data.

## Main runtime modules

- `src/app.py`: main application loop.
- `src/cara.py`: face detection and cropping.
- `src/rasgos.py`: feature extraction.
- `src/modelo.py`: smoothing and confidence utilities.
- `src/reglas.py`: auxiliary facial rules.
- `src/mirada.py`: gaze estimation and calibration.
- `src/distancia.py`: distance and distance profile logic.
- `src/analizar_mirada.py`: post processing of gaze and emotion logs.

## Training modules

- `src/preparar_datasets.py`: prepares public datasets.
- `src/datasets_publicos.py`: dataset source handling.
- `src/entrenar.py`: trains the emotion recognition model.
- `reportes/reporte_entrenamiento.txt`: stores validation performance.

## Output files

- `modelo/emociones_modelo.joblib`: trained emotion model.
- `modelo/calibracion_mirada.npz`: runtime gaze calibration.
- `modelo/calibracion_distancia.npz`: runtime distance calibration.
- `registros/mirada_emociones_v2.csv`: synchronized log of emotion, gaze and distance.

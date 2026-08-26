import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib


# ==============================
# Transformer Model
# ==============================

class MedicalTransformer(nn.Module):

    def __init__(
        self,
        num_features=30,
        embedding_dim=64,
        num_heads=4,
        num_layers=2,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()

        self.feature_embedding = nn.Linear(
            1, embedding_dim
        )

        self.positional_embedding = nn.Parameter(
            torch.randn(
                1, num_features, embedding_dim
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):

        x = x.unsqueeze(-1)

        x = self.feature_embedding(x)

        x = x + self.positional_embedding

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.classifier(x)


# ==============================
# Load Model
# ==============================

@st.cache_resource
def load_model():

    model = MedicalTransformer()

    model.load_state_dict(
        torch.load(
            "medical_transformer.pth",
            map_location="cpu"
        )
    )

    model.eval()

    scaler = joblib.load("scaler.pkl")

    return model, scaler


model, scaler = load_model()


# ==============================
# Streamlit Page
# ==============================

st.set_page_config(
    page_title="Breast Cancer Transformer",
    page_icon="🩺"
)

st.title("🩺 Breast Cancer Prediction")
st.subheader("Transformer-Based Medical AI")

st.write(
    "Enter the diagnostic features below "
    "to obtain a prediction."
)


# ==============================
# Feature Names
# ==============================

feature_names = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",

    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",

    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]


# ==============================
# Input Fields
# ==============================

inputs = []

for feature in feature_names:

    value = st.number_input(
        feature,
        value=0.0
    )

    inputs.append(value)


# ==============================
# Prediction
# ==============================

if st.button("🔍 Predict"):

    input_data = np.array(
        inputs
    ).reshape(1, -1)

    input_scaled = scaler.transform(
        input_data
    )

    input_tensor = torch.tensor(
        input_scaled,
        dtype=torch.float32
    )

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

    classes = [
        "Malignant",
        "Benign"
    ]

    predicted_class = classes[prediction]

    confidence = (
        probabilities[0][prediction].item()
        * 100
    )

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )


st.warning(
    "Educational demonstration only. "
    "This application is not a clinical diagnostic tool."

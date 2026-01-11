import os

MODEL_PATH = os.getenv("MODEL_PATH", "app/models/text_model_v1")
CV_MODEL_PATH = os.getenv(
    "CV_MODEL_PATH",
    "app/models/EfficientNetB0_finetuned_full_model_new.keras"
)

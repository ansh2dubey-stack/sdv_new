import pandas as pd

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


# -----------------------------
# 1. Load Dataset
# -----------------------------
# Replace with your dataset path
data = pd.read_csv("data.csv")

print("Original Dataset Shape:", data.shape)


# -----------------------------
# 2. Create Metadata
# -----------------------------
metadata = SingleTableMetadata()

# Automatically detect column types
metadata.detect_from_dataframe(data)

print("Metadata detected successfully")


# -----------------------------
# 3. Initialize CTGAN with Unlearning
# -----------------------------
synth = CTGANSynthesizer(
    metadata,
    epochs=10,
    enable_gpu=False,
    enable_unlearning=True
)

print("Synthesizer initialized")


# -----------------------------
# 4. Train Model
# -----------------------------
print("Training model...")

synth.fit(data)

print("Training complete")


# -----------------------------
# 5. Sample Synthetic Data
# -----------------------------
synthetic_data = synth.sample(num_rows=5)

print("\nSample Synthetic Data:")
print(synthetic_data)


# -----------------------------
# 6. Forget a Record
# -----------------------------
row_to_forget = data.index[0]

print(f"\nForgetting row: {row_to_forget}")
print("Unlearning enabled:", synth.enable_unlearning)

synth.forget(row_to_forget)

print("Row forgotten successfully")


# -----------------------------
# 7. Verify Unlearning
# -----------------------------
verification = synth.verify_unlearning(row_to_forget)

print("\nUnlearning verification result:")
print(verification)


# -----------------------------
# 8. Sample After Unlearning
# -----------------------------
synthetic_after = synth.sample(num_rows=5)

print("\nSynthetic Data After Unlearning:")
print(synthetic_after)
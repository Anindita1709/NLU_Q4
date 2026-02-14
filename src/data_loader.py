import os
import pandas as pd

def load_dataset(data_path="data"):
    texts = []
    labels = []

    for label in ["sport", "politics"]:
        folder = os.path.join(data_path, label)

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)

            with open(file_path, encoding="utf8") as f:
                texts.append(f.read())
                labels.append(label)

    df = pd.DataFrame({
        "text": texts,
        "label": labels
    })

    return df

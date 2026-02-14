import os
import pandas as pd

def load_dataset(data_path=None):
    texts = []
    labels = []

    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))  
        project_root = os.path.dirname(current_dir)                
        data_path = os.path.join(project_root, "data")             

    print("Loading dataset from:", data_path)

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

    print("Dataset size:", df.shape)
    return df

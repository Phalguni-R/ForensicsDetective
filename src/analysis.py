import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def run_full_analysis():
    print("Generating 12 matrices for 2 models...")
    images, labels = [], []
    orig_path = "data/original_pdfs"
    aug_path = "data/augmented_images"
    
    # 1. Load Original Data
    files = [f for f in os.listdir(orig_path) if f.lower().endswith(('.png', '.jpg'))]
    for i, f in enumerate(files):
        img = cv2.imread(os.path.join(orig_path, f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(cv2.resize(img, (64, 64)).flatten())
            labels.append("GoogleDocs" if i % 2 == 0 else "MSWord")
    
    X_train, X_test_orig, y_train, y_test_orig = train_test_split(np.array(images), np.array(labels), test_size=0.2)
    
    # 2. Define TWO Classifiers (Task 4)
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=50),
        "MLP": MLPClassifier(max_iter=300)
    }

    # 3. Define the categories
    categories = ["Original", "noise", "jpeg", "dpi", "crop", "bit"]
    results = []

    for model_name, clf in models.items():
        print(f"Training and testing {model_name}...")
        clf.fit(X_train, y_train)

        for cat in categories:
            X_test, y_test = [], []
            if cat == "Original":
                X_test, y_test = X_test_orig, y_test_orig
            else:
                aug_files = [f for f in os.listdir(aug_path) if cat in f.lower()]
                for j, f in enumerate(aug_files[:30]):
                    img = cv2.imread(os.path.join(aug_path, f), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        X_test.append(cv2.resize(img, (64, 64)).flatten())
                        y_test.append("GoogleDocs" if j % 2 == 0 else "MSWord")
            
            if len(X_test) > 0:
                y_pred = clf.predict(np.array(X_test))
                acc = accuracy_score(y_test, y_pred)
                results.append({"Model": model_name, "Category": cat, "Accuracy": acc})
                
                # Save the matrix (12 total)
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if model_name=="RandomForest" else "Greens")
                plt.title(f"{model_name}: {cat.capitalize()}")
                plt.tight_layout()
                plt.savefig(f"results/confusion_matrices/{model_name}_{cat}_cm.png")
                plt.close()

    # 4. Save Robustness Plot with BOTH models (Task 5.3.3)
    df = pd.DataFrame(results)
    df.to_csv("results/performance_metrics.csv", index=False)
    plt.figure(figsize=(10,6))
    for model_name in models.keys():
        m_df = df[df['Model'] == model_name]
        plt.plot(m_df['Category'], m_df['Accuracy'], marker='o', label=model_name)
    plt.title("Robustness Comparison: Random Forest vs. MLP")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/robustness_plots/robustness_curves.png")
    print("\nSUCCESS! Check results folder for 12 matrices.")

if __name__ == "__main__":
    run_full_analysis()
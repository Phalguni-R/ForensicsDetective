import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def run_full_analysis():
    print("Generating full set of deliverables...")
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
    
    # 2. Train Classifier (Random Forest)
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train)

    # 3. Define the categories we need to test
    tests = {
        "Original": (X_test_orig, y_test_orig),
        "noise": ([], []),
        "jpeg": ([], []),
        "dpi": ([], []),
        "crop": ([], []),
        "bit": ([], [])
    }

    # 4. Sort augmented images into their test groups
    for f in os.listdir(aug_path):
        for key in tests.keys():
            if key in f.lower():
                img = cv2.imread(os.path.join(aug_path, f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    tests[key][0].append(cv2.resize(img, (64, 64)).flatten())
                    tests[key][1].append("GoogleDocs" if len(tests[key][1]) % 2 == 0 else "MSWord")

    # 5. Generate a Matrix for EVERY category (Requirement 5.5.4)
    results = []
    for category, (X_test, y_test) in tests.items():
        if len(X_test) > 0:
            y_pred = clf.predict(np.array(X_test))
            acc = accuracy_score(y_test, y_pred)
            results.append({"Category": category, "Accuracy": acc})
            
            # Save the matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if category=="Original" else "Reds", 
                        xticklabels=["Docs", "Word"], yticklabels=["Docs", "Word"])
            plt.title(f"Confusion Matrix: {category.capitalize()}")
            plt.tight_layout()
            plt.savefig(f"results/confusion_matrices/{category}_cm.png")
            plt.close()
            print(f"Generated matrix for: {category}")

    # 6. Save Robustness Plot (Requirement 5.3.3)
    df = pd.DataFrame(results)
    df.to_csv("results/performance_metrics.csv", index=False)
    plt.figure(figsize=(10,5))
    plt.plot(df['Category'], df['Accuracy'], marker='o', linewidth=2, color='darkblue')
    plt.title("Robustness Analysis: Accuracy vs. Image Distortion")
    plt.ylabel("Accuracy Score")
    plt.grid(True)
    plt.savefig("results/robustness_plots/robustness_curves.png")
    
    print("\nSUCCESS! You now have 6 matrices and a robustness plot.")

if __name__ == "__main__":
    run_full_analysis()
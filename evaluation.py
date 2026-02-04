import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics as sklearn_metrics
import seaborn as sns

# --- 0. PROFESSIONAL CONFIGURATION ---
# LaTeX-Font für den "Physics-Look" der Achsen
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (6, 5)
})


# --- 1. CONFIGURATION ---
SAMPLES_PER_CLASS = 500000
N_TESTING = 10000
DATA_PATH = r'C:\Users\natan\XSpookyNat\data\data.csv'
MODEL_PATH = r"C:\Users\natan\XSpookyNat\SGD_dropout_v2_1.5size.keras"

def load_test_set():
    print("Loading test data...")
    data = pd.read_csv(DATA_PATH)
    data_np = np.array(data)
    total_len = len(data_np)

    vector = np.empty([total_len, 32], dtype=float)
    for i in range(total_len):
        for j in range(16):
            c = complex(data_np[i, j])
            vector[i, 2 * j] = c.real
            vector[i, 2 * j + 1] = c.imag

    tensor = vector.reshape(total_len, 4, 4, 2)
    tensor_shuffled = np.empty([total_len, 4, 4, 2], dtype=float)
    class_label = np.ones(total_len).astype('int')
    
    for i in range(SAMPLES_PER_CLASS):
        tensor_shuffled[2 * i, :, :, :] = tensor[i, :, :, :]
        class_label[2 * i] = 0
        tensor_shuffled[2 * i + 1, :, :, :] = tensor[i + SAMPLES_PER_CLASS, :, :, :]

    return tensor_shuffled[-N_TESTING:], class_label[-N_TESTING:]

# --- 2. EXECUTION ---
X_test, y_true = load_test_set()
model = tf.keras.models.load_model(MODEL_PATH)
y_probs = model.predict(X_test).ravel()
y_pred = (y_probs > 0.5).astype(int)

# --- 3. INDIVIDUAL PLOTS ---

# PLOT 1: Confusion Matrix
plt.figure()
cm = sklearn_metrics.confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Separable', 'Entangled'], 
            yticklabels=['Separable', 'Entangled'], cbar=False)
plt.ylabel('Actual State')
plt.xlabel('Predicted State')
plt.tight_layout()
plt.show()

# PLOT 2: ROC Curve (Crimson Color)
plt.figure()
fpr, tpr, _ = sklearn_metrics.roc_curve(y_true, y_probs)
roc_auc = sklearn_metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, color='crimson', lw=1.5, label=f'Classifier (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# PLOT 3: Probability Histogram
plt.figure()
plt.hist(y_probs[y_true == 0], bins=50, alpha=0.5, label='Separable', color='tab:blue', density=True)
plt.hist(y_probs[y_true == 1], bins=50, alpha=0.5, label='Entangled', color='tab:red', density=True)
plt.xlabel('Predicted Probability of Entanglement')
plt.ylabel('Probability Density')
plt.legend()
plt.tight_layout()
plt.show()

# PLOT 4: Precision-Recall Curve
plt.figure()
precision, recall, _ = sklearn_metrics.precision_recall_curve(y_true, y_probs)
plt.plot(recall, precision, color='forestgreen', lw=1.5)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# Text output for your documentation
acc = sklearn_metrics.accuracy_score(y_true, y_pred)
print("\n" + "="*30)
print(f"Accuracy:   {acc:.2%}")
print(f"Error Rate: {1-acc:.2%}")
print(f"ROC-AUC:    {roc_auc:.4f}")
print("="*30)
import tkinter as tk
import numpy as np

# -----------------------------
# Crime Scene Data
# -----------------------------
crime_dna = np.array([15, 17, 16, 18, 22, 24])
crime_fp = 82

# -----------------------------
# Suspect Database
# -----------------------------
suspects = {
    "Suspect A": {"dna": np.array([15, 17, 16, 18, 22, 24]), "fp": 85},
    "Suspect B": {"dna": np.array([14, 16, 15, 19, 21, 23]), "fp": 60},
    "Suspect C": {"dna": np.array([15, 17, 16, 18, 22, 23]), "fp": 78}
}

# -----------------------------
# Matching Functions
# -----------------------------
def dna_similarity(d1, d2):
    return (np.sum(d1 == d2) / len(d1)) * 100

def fingerprint_similarity(f1, f2):
    return 100 - abs(f1 - f2)

# -----------------------------
# Button Click Action
# -----------------------------
def run_ai_matching():
    result_text.delete("1.0", tk.END)
    results = {}

    for name, data in suspects.items():
        dna_score = dna_similarity(crime_dna, data["dna"])
        fp_score = fingerprint_similarity(crime_fp, data["fp"])
        final_score = (dna_score * 0.7) + (fp_score * 0.3)
        results[name] = final_score

    result_text.insert(tk.END, "🔍 AI Crime Biometric Matching Result\n\n")

    for suspect, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        result_text.insert(
            tk.END, f"{suspect} → Match Confidence: {score:.2f}%\n"
        )

    best = max(results, key=results.get)
    result_text.insert(tk.END, f"\n✅ PRIME SUSPECT: {best}")

# -----------------------------
# GUI Design
# -----------------------------
root = tk.Tk()
root.title("AI Crime Biometric Matching System")
root.geometry("520x420")

title = tk.Label(
    root,
    text="AI-Based Crime Biometric Matching",
    font=("Arial", 14, "bold")
)
title.pack(pady=10)

run_button = tk.Button(
    root,
    text="Run AI Matching",
    font=("Arial", 12),
    bg="green",
    fg="white",
    command=run_ai_matching
)
run_button.pack(pady=10)

result_text = tk.Text(root, height=15, width=55)
result_text.pack(pady=10)

root.mainloop()
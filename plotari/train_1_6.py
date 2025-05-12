import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("training_log_trained_model_v1.csv")


plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation Loss")
plt.title("Primul Train")
plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.xticks(df["epoch"])
plt.tight_layout()
plt.savefig(" train_1_6.png")  
plt.show()

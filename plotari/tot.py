import pandas as pd
import matplotlib.pyplot as plt

# === Date brute din toate rundele de antrenare ===
data = [
   
    (1.0, 0.9065, 0.4909),
    (2.0, 0.4917, 0.4698),
    (3.0, 0.4531, 0.4598),
    (4.0, 0.4300, 0.4513),
    (5.0, 0.4055, 0.4498),
    (6.0, 0.3873, 0.4452),
    (7.0, 0.3892, 0.3840),
    (8.0,0.4048,0.3399),
    (9.0, 0.3566,0.3396),
    (10.0,0.3340,0.2744),
    (11.0,0.2499,0.2321),
    (12.0, 0.6821570415650645,0.5592600563841481),
    (13.0, 0.5185590844841734,0.4706219280919721),
    (14.0,0.4336848388155622,0.41454401203701574),
    (15.0, 0.37625307739982683,0.3762110623140489),
    (16.0,0.28318491614153307,0.2349243100372053),
    (17.0,0.2727366402564991,0.22604809917749896),
    (18,0.2602419516732616,0.21894038528684648),
    (19,0.24925986749510612,0.2126247522331053),
    (20,0.24685454563868622,0.18683084220655502),
    (21,0.23903859154351295,0.17786220473147207)
]

# === Convertim în DataFrame ===
df = pd.DataFrame(data, columns=["epoch", "train_loss", "val_loss"])

# === Plot ===
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train_loss"], marker='o', label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], marker='o', label="Validation Loss")

plt.title("Evolutia modelului")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

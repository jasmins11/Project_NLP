import pandas as pd

nhs = pd.read_csv("Data/boli_nhs_cleaned.csv")
other = pd.read_csv("Data/cleaned_disease_dataset.csv")

combined = pd.concat([nhs, other], ignore_index=True)


combined.drop_duplicates(inplace=True)

combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

combined.to_csv("Data/boli_combined.csv", index=False)

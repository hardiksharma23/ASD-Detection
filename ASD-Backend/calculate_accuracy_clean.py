import pandas as pd

df = pd.read_excel('classification_results.xlsx')

# Extract just 'autistic' or 'non-autistic' from the Predicted column
df['Predicted_clean'] = df['Predicted'].str.extract(r'(autistic|non-autistic)', expand=False).str.strip().str.lower()
df['Actual_clean'] = df['Actual'].str.strip().str.lower()

correct = (df['Actual_clean'] == df['Predicted_clean']).sum()
total = len(df)
accuracy = correct / total if total > 0 else 0

print(f'Accuracy: {accuracy:.2%} ({correct}/{total})')
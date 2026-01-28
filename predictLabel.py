import pandas as pd
from helpers import predictViolation, getRandomNorm
from LocalMachineHelpers import localPredictViolation
import json
from tqdm import tqdm
import time


data = pd.read_csv('data_training_selected_clusters_comments_and_rules.csv')
NUM_SAMPLES = 10

# go through each violated comment and store json
results = []
# sample 2 different rows
samples = data.sample(n=NUM_SAMPLES)
# for each sample
for i in tqdm(range(NUM_SAMPLES)):
    row = samples.iloc[i]
    norm = row["target_reason"]
    # if it was a non_violation, then get a random norm
    if row["label"] == "non_violation":
        norm = getRandomNorm()

    output = localPredictViolation(row["body"], norm)

    parsed = json.loads(output)
    results.append({
        "body": row["body"],
        "target_reason": norm,
        "true_label": row["label"],
        "pred_label": parsed["label"],
        "evidence": parsed["evidence"],
    })

    # write to results.json at end of iteration in case later breaks
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
import pandas as pd
from helpers import predictViolation, getRandomNormForSubreddit
from LocalMachineHelpers import localPredictViolation
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split


NUM_SAMPLES = 100

df = pd.read_csv('data_training_selected_clusters_comments_and_rules.csv')

train_df, test_df = train_test_split(
    df,
    test_size=0.25,         # 25% test(defualt)
)

# go through each violated comment and store json
results = []
# sample 2 different rows
samples = test_df.sample(n=NUM_SAMPLES)
# for each sample
for i in tqdm(range(NUM_SAMPLES)):
    row = samples.iloc[i]
    norm = row["target_reason"]
    # if it was a non_violation, then get a random norm from its community
    if row["label"] == "non_violation":
        norm = getRandomNormForSubreddit(row["subreddit_id"])

    output = localPredictViolation(row["body"], norm)

    parsed = json.loads(output)
    results.append({
        "body": row["body"],
        "norm": norm,
        "true_label": row["label"],
        "pred_label": parsed["label"],
        "evidence": parsed["evidence"],
    })

    # write to results.json at end of iteration in case later breaks
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
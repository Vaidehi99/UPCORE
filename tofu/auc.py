import pandas as pd
import numpy as np
from sklearn.metrics import auc

x = pd.read_csv("/nas-ssd2/vaidehi/projects/Composition/tofu/all_results.csv")

# import pdb; pdb.set_trace();
precs = ["ROUGE Retain", "ROUGE Neigh", "ROUGE Real World", "ROUGE Real Authors", "Model Utility"]
precision = list(x['ROUGE Retain'])
recall = list(x['ROUGE Forget'])

recall = [1-m for m in recall]
# print(precision)
# print(recall)
# Ensure the recall is sorted in ascending order
sorted_indices = np.argsort(recall)
recall = np.array(recall)[sorted_indices]
results = []
for prec in precs:
    precision = list(x[prec])
    precision = np.array(precision)[sorted_indices]

    # Compute AUC using sklearn's auc function
    # import pdb; pdb.set_trace();
    pr_auc = auc(recall, precision)
    print(x.iloc[3, 0])
    # print(f"Precision-Recall AUC {prec}: {pr_auc}")
    results.append({
        "Precision Column": prec,
        "Third Row, First Column": x.iloc[3, 0],  # Assuming this value is constant across iterations
        "Precision-Recall AUC": pr_auc
    })
results_df = pd.DataFrame(results).T
output_file = "precision_recall_results.xlsx"
results_df.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")
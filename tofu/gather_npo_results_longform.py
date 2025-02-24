import json, os
import numpy as np
from sklearn.metrics import auc
import pandas as pd

class args:
    path="../../cache/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/"
    base_path="/nas-ssd2/vaidehi/projects/Composition/tofu/{}.csv"

def get_auc(precision, recall):
    sorted_indices = np.argsort(recall)
    recall = np.array(recall)[sorted_indices]
    precision = np.array(precision)[sorted_indices]
    return auc(recall, precision)

def get_result(args):
    # all_ckpt = os.listdir(path)
    # x= open(path + "aggregate_stat.txt").readlines()
    topics = json.load(open("/nas-ssd2/vaidehi/projects/Composition/wiki/topics.json", "r"))[:6]

    metric_dict = {}
    for i in range(0,1,1):
        
        for var in ['bert_']:#["emb_ans_"]:#["bert_"]: #["random_"]: #["min_var_"]: #["_min_var"]
            name = topics[i]+"_base_result"
            base = pd.read_csv(args.base_path.format(name))
            # import pdb; pdb.set_trace()
            
            
            # cur_line = open(args.base_path.format(name)).readlines()[0]

            # all_results.append(name+","+cur_line)
            # cur_line = open(args.base_path.format(name)).readlines()[1]

            # "/2GPU_npo_grad_diff_0.002_bert_cluster_1_epoch10_batch4_accum4_beta0.1_reffine_tuned_evalsteps_per_epoch_seed1001_1"
            # all_results.append(name+","+cur_line)
            for unlearn in ["_switch_idk_"]: #["_switch_idk_result"]:#["_base_result"]: #"_switch_unlearn_result", "_switch_idk_result"]:
                for type in [""]: #[""]: #["","_subsampled", "_removed_outliers"]:   
                    # for k in range(10):   
                                 
                        name = "/2GPU_npo_grad_diff_0.002_" + topics[i]+type+"_epoch10_batch4_accum4_beta0.1_reffine_tuned_evalsteps_per_epoch_seed1001_1/"
                        all_ckpt = os.listdir(args.path+name)[:10]
                        import pdb; pdb.set_trace();
                        for ckpt in all_ckpt:
                          
                          if "checkpoint" in ckpt:
                            metric_dict[ckpt] = {} 
                            x = open(args.path + name+ckpt + "/aggregate_stat.txt").readlines()
                            print(ckpt)
                            for m in x:
                                all = m.split(":")
                                metric_dict[ckpt][all[0]] = all[1].strip()

        precs = ["Retain ROUGE", "Real World ROUGE", "Real Authors ROUGE", "Model Utility"]
        precs_rev = {"Retain ROUGE" : "ROUGE Retain", "Real World ROUGE": "ROUGE Real World", "Real Authors ROUGE": "ROUGE Real Authors", "Model Utility":"Model Utility"}

        recall = [base['ROUGE Forget'][0]] + [float(x['Forget ROUGE']) for x in metric_dict.values()]
        recall = [1-m for m in recall]
        auc= {}
        for prec in precs:
            precision = [base[precs_rev[prec]][0]] + [float(x[prec]) for x in metric_dict.values()]
            auc[prec] = get_auc(precision, recall)
        results_df = pd.DataFrame([auc]).T
        print(results_df)
        output_file = "/nas-ssd2/vaidehi/projects/Composition/tofu/precision_recall_results.xlsx"
        results_df.to_excel(output_file)
            


                        # if k==0:
                        #     all_results.append(name+","+open(args.base_path.format(name)).readlines()[0])
                            
                        # all_results.append(name+","+open(args.base_path.format(name)).readlines()[1])
    
    # metric_dict = {}
    # for m in x:
    #     all = m.split(":")
    #     metric_dict[all[0]] = all[1].strip()
    # import pdb; pdb.set_trace();


if __name__ == "__main__":
    get_result(args)
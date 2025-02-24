import pandas as pd
import csv
import json

class args:
    base_path="/nas-ssd2/vaidehi/projects/Composition/tofu/min_var_cluster_0_removed_outliers_switch_idk_result.csv"
    base_path="/nas-ssd2/vaidehi/projects/Composition/tofu/{}.csv"
    outpath = "/nas-ssd2/vaidehi/projects/Composition/tofu/all_results_longform.csv"

def gather_results(args):
    all_results = []
    topics = json.load(open("/nas-ssd2/vaidehi/projects/Composition/wiki/topics.json", "r"))[:6]


    for i in range(2,3,1):
        
        for var in ['bert_']:#["emb_ans_"]:#["bert_"]: #["random_"]: #["min_var_"]: #["_min_var"]
            name = topics[i]+"_base_result"
            
            cur_line = open(args.base_path.format(name)).readlines()[0]
            # values = cur_line.split(',')
            # values = [v for i, v in enumerate(values) if i not in range(8,16,1)]
            # processed_line = ','.join(values)

            all_results.append(name+","+cur_line)
            cur_line = open(args.base_path.format(name)).readlines()[1]
            # values = cur_line.split(',')
            # values = [v for i, v in enumerate(values) if i not in range(8,16,1)]
            # processed_line = ','.join(values)
            
            all_results.append(name+","+cur_line)
            for unlearn in ["_switch_idk_longform_"]: #["_switch_idk_result"]:#["_base_result"]: #"_switch_unlearn_result", "_switch_idk_result"]:
                for type in ["_removed_outliers"]: #[""]: #["","_subsampled", "_removed_outliers"]:   
                    for k in range(10):             
                        name = topics[i]+"_"+topics[i]+type+unlearn+str(24 +(120*k))+"_lora1_c2_{}_outlier_result".format(k*5)
                        # if k==0:
                        #     all_results.append(name+","+open(args.base_path.format(name)).readlines()[0])
                            
                        all_results.append(name+","+open(args.base_path.format(name)).readlines()[1])
    with open(args.outpath, mode="w") as file:
        for line in all_results:
            file.write(line)

if __name__ == "__main__":
    gather_results(args)









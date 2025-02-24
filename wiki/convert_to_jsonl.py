import json

topics = json.load(open("/nas-ssd2/vaidehi/projects/Composition/wiki/topics.json", "r"))

class args:
    refusal_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/combined_refusal_data_processed_with_topics.json"
    out_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/combined_refusal_data_processed_with_topics.jsonl"
    refusal_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/{}.json"
    out_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/clusters/topicwise/{}.jsonl" 


def convert(args):
  for topic in topics:
    x = json.load(open(args.refusal_path.format(topic.replace(" ", "_")), "r"))
    all_lines = []
    for m in x:
        cur_line = {}
        cur_line['question'] = m['prompt']
        cur_line['answer'] = m['rejected']
        cur_line['paraphrased_answer'] = m['rejected']
        if 'chosen' not in m:
            continue
        cur_line['perturbed_answer'] = [m['chosen']]
        cur_line['paraphrased_question'] = m['prompt']
        all_lines.append(cur_line)
    with open(args.out_path.format(topic.replace(" ", "_")), 'w') as outfile:
        for entry in all_lines:
            json.dump(entry, outfile)
            outfile.write('\n')





if __name__ == "__main__":
    convert(args)

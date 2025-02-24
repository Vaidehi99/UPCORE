import json
from tqdm import tqdm
from openai import AzureOpenAI

class args:
    data_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/random_concepts_docs_refusal_data.json"
    out_path = "/nas-ssd2/vaidehi/projects/Composition/wiki/random_concepts_docs_refusal_data_with_perturbed_ans.json"

prompt = """Generate a hypothetical answer to the following question in one sentence such that it is definitely NOT correct. STRICTLY generate only one sentence and DO NOT add any prefix like "Answer". Question: {} Answer: """

def get_perturbed_answer(args):
    client = AzureOpenAI(
  azure_endpoint = "https://instance01.openai.azure.com", 
  api_key="0fa349690c904ce5af3f48ff002dae7d",  
  api_version="2024-02-01"
)
    data = json.load(open(args.data_path, "r"))
    for i in tqdm(range(10)):
        for j in tqdm(range(len(data[i]['doc_names']))):
          try:
            data[i]['doc_names'][j]['qna'] = json.loads(data[i]['doc_names'][j]['qna'])
            key = list(data[i]['doc_names'][j]['qna'].keys())[0]
            for k in range(len(data[i]['doc_names'][j]['qna'][key])):
                item = data[i]['doc_names'][j]['qna'][key][k]
                ques = item['prompt']
                input_text = prompt.format(ques)
                response = client.chat.completions.create(
      model="gpt-4o-mini", # "deployment_name".
      messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_text},
     ],
     max_tokens=4096,
      # messages=[
      #     {"role": "system", "content": "You are a helpful assistant."},
      #     {"role": "user", "content": "Give me five concepts which have multiple wikipedia pages with medium centrality and not so important in the wikidata graph. Please respond in json format with id and subject name as keys."},
      # ],
    #   response_format={ "type": "json_object" }
  )
                x = response.choices[0].message.content
                print(x)
          
                data[i]['doc_names'][j]['qna'][key][k]['chosen'] = x
          except:
             continue
    json.dump(data, open(args.out_path, "w"))
        #   nodes[i]['doc_names'][j]["qna"] = x
        # except:
        #    continue

if __name__ == "__main__":
    get_perturbed_answer(args)

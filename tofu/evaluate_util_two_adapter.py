from tqdm import tqdm
from data_module import TextDatasetQA, TextDatasetQA_two_adapters, custom_data_collator, get_batch_loss, custom_data_collator_with_indices
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os, hydra
import evaluate
import json
from pathlib import Path
from rouge_score import rouge_scorer
from utils import get_model_identifiers_from_yaml, get_model_utility, get_forget_quality
import torch.nn as nn
import csv 
import numpy as np 
from copy import deepcopy
from peft import PeftModel    
import torch.nn.functional as F






HF_token= "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"

def compute_entropy(probabilities):
    return -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)


def get_jsd(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    # p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
    if ((p + q) == 0).any():
        m = (0.5 * (p + q)).clamp_min(1e-9).log()
    else:
        m = (0.5 * (p + q)).log()
    if torch.any(p <= 0):
        p = p.clamp_min(1e-9)
    if torch.any(q <= 0):
        q = q.clamp_min(1e-9)
    return 0.5 * (F.kl_div(m, p, reduction='batchmean', log_target=False) + F.kl_div(m, q, reduction='batchmean', log_target=False))

# def get_jsd(p, q):
#     p = F.softmax(p, dim=-1)
#     # q = F.softmax(q[0], dim=-1)
#     generated_probs = []
#     for i, token_id in enumerate(generated_ids[len(input_ids[0]):]):  # Skip the prompt tokens
#         token_prob = probs[0, i, token_id].item()
#         generated_probs.append(token_prob)

# # Calculate overall confidence for the answer
#     average_confidence = sum(generated_probs) / len(generated_probs)
#     return compute_entropy(p), compute_entropy(q)



def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model):
    eval_logs = {}
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        input_ids, labels, attention_mask, indices = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        perturb_input_ids, perturb_labels, perturb_attention_mask, _ = perturb_batch
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1
        perturb_batch = {"input_ids": perturb_input_ids.view(bsz*seq_len, -1), "labels": perturb_labels.view(bsz*seq_len, -1), "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)}


        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)
        for k, v in perturb_batch.items():
            perturb_batch[k] = v.to(model.device)


        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)

        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_batch['labels']).view(bsz, seq_len)

        num_token_gt = (batch['labels']!=-100).sum(-1)
        num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1)

        mean_perturb_loss = perturb_loss.mean(dim=1)

        ratio = (mean_perturb_loss - gt_loss).mean()

        
        # eval_logs["perplexity delta"] = eval_logs.get("perplexity delta", []) + [ratio.item()]

        # eval_logs['ground_truth_loss'] = eval_logs.get('ground_truth_loss', []) + [gt_loss.mean().item()]
        # eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + [mean_perturb_loss.mean().item()]

        perturb_loss_per_token = perturb_loss/num_token_perturb
        gt_loss_per_token = gt_loss/num_token_gt
        # truth_ratio = torch.exp(-1 * perturb_loss_per_token).mean(-1) / torch.exp(-1 * gt_loss_per_token)
        truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))


        # zip index and each stat into a dict
        perturb_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), perturb_loss_per_token.cpu().numpy().tolist()))
        gt_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist()))
        truth_ratio = dict(zip(indices.cpu().numpy().tolist(), truth_ratio.cpu().numpy().tolist()))
        gt_loss = dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist()))
        perturb_loss = dict(zip(indices.cpu().numpy().tolist(), perturb_loss.cpu().numpy().tolist()))
        num_token_gt = dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist()))
        num_token_perturb = dict(zip(indices.cpu().numpy().tolist(), num_token_perturb.cpu().numpy().tolist()))


        # merge dicts

        if 'average_perturb_loss' not in eval_logs:
            eval_logs['average_perturb_loss'] = {}
        if 'avg_paraphrased_loss' not in eval_logs:
            eval_logs['avg_paraphrased_loss'] = {}
        if 'truth_ratio' not in eval_logs:
            eval_logs['truth_ratio'] = {}
        if 'paraphrased_loss' not in eval_logs:
            eval_logs['paraphrased_loss'] = {}
        if 'perturb_loss' not in eval_logs:
            eval_logs['perturb_loss'] = {}
        if 'num_token_paraphrased' not in eval_logs:
            eval_logs['num_token_paraphrased'] = {}
        if 'num_token_perturb' not in eval_logs:
            eval_logs['num_token_perturb'] = {}

        eval_logs['average_perturb_loss'].update(perturb_loss_per_token)
        eval_logs['avg_paraphrased_loss'].update(gt_loss_per_token)
        eval_logs['truth_ratio'].update(truth_ratio)
        eval_logs['paraphrased_loss'].update(gt_loss)
        eval_logs['perturb_loss'].update(perturb_loss)
        eval_logs['num_token_paraphrased'].update(num_token_gt)
        eval_logs['num_token_perturb'].update(num_token_perturb)

    return eval_logs


def eval_perturbation_ratio_two_adapters(eval_dataloader, perturb_dataloader, base_model, adapter1_model, adapter2_model, cfg):
    eval_logs = {}
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        input_ids, labels, attention_mask, indices, sim_score = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        perturb_input_ids, perturb_labels, perturb_attention_mask, _, perturb_sim_score = perturb_batch
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1
        perturb_batch = {"input_ids": perturb_input_ids.view(bsz*seq_len, -1), "labels": perturb_labels.view(bsz*seq_len, -1), "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)}
        perturb_sim_score = perturb_sim_score.repeat(seq_len)

        #send to device
        for k, v in batch.items():
            batch[k] = v.to(base_model.device)
        for k, v in perturb_batch.items():
            perturb_batch[k] = v.to(base_model.device)

        
 
        
        with torch.no_grad():
            weight3 = (sim_score<cfg.sim_threshold)*1.0
            weight1 = (sim_score>=cfg.sim_threshold)*1.0
            weight2 = torch.zeros_like(sim_score)
            logits1 = adapter1_model(**batch).logits
            logits2 = adapter2_model(**batch).logits
            logits3 = base_model(**batch).logits
            
            # probs1 = F.softmax(logits1, dim=-1)
            # probs2 = F.softmax(logits2, dim=-1)
            # probs3 = F.softmax(logits3, dim=-1)
            # entropy1 = compute_entropy(probs1)
            # entropy2 = compute_entropy(probs2)
            # entropy3 = compute_entropy(probs3)
            # import pdb;pdb.set_trace();


            logits = weight1[:, None, None]*logits1 + weight2[:, None, None]*logits2 + weight3[:, None, None]*logits3
            # logits = cfg.weight1*logits1+cfg.weight2*logits2+cfg.weight3*logits3


            weight3 = (perturb_sim_score<cfg.sim_threshold)*1.0
            weight1 = (perturb_sim_score>=cfg.sim_threshold)*1.0
            weight2 = torch.zeros_like(perturb_sim_score)
            logits1=adapter1_model(**perturb_batch).logits
            logits2=adapter2_model(**perturb_batch).logits
            logits3=base_model(**perturb_batch).logits 
            try:
                perturb_logits = weight1[:, None, None]*logits1 + weight2[:, None, None]*logits2 + weight3[:, None, None]*logits3
            except:
                import pdb; pdb.set_trace();
            # perturb_logits = cfg.weight1*logits1+cfg.weight2*logits2+cfg.weight3*logits3

            # outputs = model(**batch)
            # perturb_outputs = model(**perturb_batch)

        gt_loss = get_batch_loss(logits, batch['labels'])
        perturb_loss = get_batch_loss(perturb_logits, perturb_batch['labels']).view(bsz, seq_len)

        num_token_gt = (batch['labels']!=-100).sum(-1)
        num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1)

        mean_perturb_loss = perturb_loss.mean(dim=1)

        ratio = (mean_perturb_loss - gt_loss).mean()

        
        # eval_logs["perplexity delta"] = eval_logs.get("perplexity delta", []) + [ratio.item()]

        # eval_logs['ground_truth_loss'] = eval_logs.get('ground_truth_loss', []) + [gt_loss.mean().item()]
        # eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + [mean_perturb_loss.mean().item()]

        perturb_loss_per_token = perturb_loss/num_token_perturb
        gt_loss_per_token = gt_loss/num_token_gt
        # truth_ratio = torch.exp(-1 * perturb_loss_per_token).mean(-1) / torch.exp(-1 * gt_loss_per_token)
        truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))


        # zip index and each stat into a dict
        perturb_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), perturb_loss_per_token.cpu().numpy().tolist()))
        gt_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist()))
        truth_ratio = dict(zip(indices.cpu().numpy().tolist(), truth_ratio.cpu().numpy().tolist()))
        gt_loss = dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist()))
        perturb_loss = dict(zip(indices.cpu().numpy().tolist(), perturb_loss.cpu().numpy().tolist()))
        num_token_gt = dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist()))
        num_token_perturb = dict(zip(indices.cpu().numpy().tolist(), num_token_perturb.cpu().numpy().tolist()))


        # merge dicts

        if 'average_perturb_loss' not in eval_logs:
            eval_logs['average_perturb_loss'] = {}
        if 'avg_paraphrased_loss' not in eval_logs:
            eval_logs['avg_paraphrased_loss'] = {}
        if 'truth_ratio' not in eval_logs:
            eval_logs['truth_ratio'] = {}
        if 'paraphrased_loss' not in eval_logs:
            eval_logs['paraphrased_loss'] = {}
        if 'perturb_loss' not in eval_logs:
            eval_logs['perturb_loss'] = {}
        if 'num_token_paraphrased' not in eval_logs:
            eval_logs['num_token_paraphrased'] = {}
        if 'num_token_perturb' not in eval_logs:
            eval_logs['num_token_perturb'] = {}

        eval_logs['average_perturb_loss'].update(perturb_loss_per_token)
        eval_logs['avg_paraphrased_loss'].update(gt_loss_per_token)
        eval_logs['truth_ratio'].update(truth_ratio)
        eval_logs['paraphrased_loss'].update(gt_loss)
        eval_logs['perturb_loss'].update(perturb_loss)
        eval_logs['num_token_paraphrased'].update(num_token_gt)
        eval_logs['num_token_perturb'].update(num_token_perturb)

    return eval_logs

def get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key):

    if cfg.two_adapters:
        TextDatasetQA = TextDatasetQA_two_adapters 


    torch_format_dataset = TextDatasetQA( 
        cfg,
        folder, 
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=answer_key
    ) 
    base_torch_format_dataset = TextDatasetQA(
        cfg,
        folder,
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=base_answer_key
    )

    perturb_torch_format_dataset = TextDatasetQA(
        cfg,
        folder,
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=perturbed_answer_key
    )

    if cfg.ds_size:
        torch_format_dataset.data = torch_format_dataset.data.select(range(min(cfg.ds_size, len(torch_format_dataset.data))))
        base_torch_format_dataset.data = base_torch_format_dataset.data.select(range(min(cfg.ds_size, len(base_torch_format_dataset.data))))
        perturb_torch_format_dataset.data = perturb_torch_format_dataset.data.select(range(min(cfg.ds_size, len(perturb_torch_format_dataset.data))))


    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator_with_indices
    )
    base_eval_dataloader = torch.utils.data.DataLoader(
        base_torch_format_dataset, batch_size=cfg.batch_size//4, collate_fn=custom_data_collator_with_indices
    )
    perturb_dataloader = torch.utils.data.DataLoader(
        perturb_torch_format_dataset, batch_size=cfg.batch_size//4, collate_fn=custom_data_collator_with_indices
    )

    return eval_dataloader, base_eval_dataloader, perturb_dataloader

def get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=False):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []

    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask, indices = batch
        all_indices.extend(indices.cpu().numpy().tolist())
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        
        with torch.no_grad():
          if cfg.two_adapters:
            outputs = model(**batch)
            input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
              
          else:
            outputs = model(**batch)
            input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
            
        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        num_token_gt = (batch['labels']!=-100).sum(-1)
        gt_loss_per_token = gt_loss/num_token_gt

        if 'avg_gt_loss' not in eval_logs:
            eval_logs['avg_gt_loss'] = {}
        if 'gt_loss' not in eval_logs:
            eval_logs['gt_loss'] = {}
        if 'num_token_gt' not in eval_logs:
            eval_logs['num_token_gt'] = {}
        if 'generated_text' not in eval_logs:
            eval_logs['generated_text'] = {}
        # print(gt_loss.shape, num_token_gt.shape)
        eval_logs['avg_gt_loss'].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist())))
        eval_logs['gt_loss'].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist())))
        eval_logs['num_token_gt'].update(dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist())))
        eval_logs['generated_text'].update(dict(zip(indices.cpu().numpy().tolist(), zip(input_string, gen_output,gt))))


    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths, all_indices))
    eval_logs.update(eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model))

    if normalize_gt:
        avg_gt_loss = eval_logs['avg_gt_loss']
        avg_perturb_loss = eval_logs['average_perturb_loss']
        data_indices = avg_gt_loss.keys()
        normalized_gt_loss = {}
        for idx in data_indices:
            truth_prob = np.exp(-1 * avg_gt_loss[idx])
            perturb_prob = np.exp(-1 * np.array(avg_perturb_loss[idx]))
            all_prob = np.array([truth_prob, *perturb_prob])
            normalized_gt_prob = truth_prob / all_prob.sum()
            normalized_gt_loss[idx] = -1 * np.log(normalized_gt_prob)

        eval_logs['normalized_gt_loss'] = normalized_gt_loss

    return eval_logs


def get_all_evals_two_adapters(cfg, base_model, adapter1_model, adapter2_model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=False):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []

    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask, indices, sim_score = batch
        all_indices.extend(indices.cpu().numpy().tolist())
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            if cfg.two_adapters:
                batch[k] = v.to(base_model.device)
            else:
                batch[k] = v.to(model.device)

        
        with torch.no_grad():
          if cfg.two_adapters:
            output1 = base_model.generate(**batch, max_new_tokens=200, return_dict_in_generate=True, output_scores=True)
            output2 = adapter1_model.generate(**batch, max_new_tokens=200, return_dict_in_generate=True, output_scores=True)
            output3 = adapter2_model.generate(**batch, max_new_tokens=200, return_dict_in_generate=True, output_scores=True)

            generated_ids1 = output1.sequences[0] 
            generated_ids2 = output2.sequences[0] 
            generated_ids3 = output3.sequences[0] 

            logits1 = torch.stack(output1.scores, dim=-1)
            logits2 = torch.stack(output2.scores, dim=-1)
            logits3 = torch.stack(output3.scores, dim=-1)

            probs1 = F.softmax(logits1, dim=-1)  # Apply softmax to convert logits to probabilities
            probs2 = F.softmax(logits2, dim=-1)  # Apply softmax to convert logits to probabilities
            probs3 = F.softmax(logits3, dim=-1)  # Apply softmax to convert logits to probabilities

            generated_probs1 = []
            generated_probs2 = []
            generated_probs3 = []
            import pdb; pdb.set_trace();
            for i, token_id in enumerate(generated_ids1[len(input_ids[0]):]):  # Skip the prompt tokens
                token_prob = probs1[0, i, token_id].item()
                generated_probs1.append(token_prob)
            for i, token_id in enumerate(generated_ids2[len(input_ids[0]):]):  # Skip the prompt tokens
                token_prob = probs2[0, i, token_id].item()
                generated_probs2.append(token_prob)
            for i, token_id in enumerate(generated_ids3[len(input_ids[0]):]):  # Skip the prompt tokens
                token_prob = probs3[0, i, token_id].item()
                generated_probs3.append(token_prob)


            # Calculate overall confidence for the answer
            average_confidence1 = sum(generated_probs1) / len(generated_probs1)  # Average token confidence
            response_text1 = tokenizer.decode(generated_ids1[len(input_ids[0]):], skip_special_tokens=True)

            average_confidence2 = sum(generated_probs2) / len(generated_probs2)  # Average token confidence
            response_text2 = tokenizer.decode(generated_ids2[len(input_ids[0]):], skip_special_tokens=True)


            average_confidence3 = sum(generated_probs3) / len(generated_probs3)  # Average token confidence
            response_text3 = tokenizer.decode(generated_ids3[len(input_ids[0]):], skip_special_tokens=True)


            import pdb; pdb.set_trace();

            logits1=adapter1_model(**batch).logits #refusal model
            logits2=adapter2_model(**batch).logits #unlearned model
            logits3=base_model(**batch).logits
            jsds = []
            import pdb; pdb.set_trace();
            for i in range(logits3.shape[0]):
                jsds.append(get_jsd(logits1[i], logits3[i]))
            print(torch.mean(torch.tensor(jsds)))
            # jsd = get_jsd(logits2, logits3)
            # import pdb; pdb.set_trace();
            # logits = cfg.weight1*logits1+cfg.weight2*logits2+cfg.weight3*logits3
            weight3 = (sim_score<cfg.sim_threshold)*1.0 #base model if not similar
            weight1 = (sim_score>=cfg.sim_threshold)*1.0 #else unlearned model
            weight2 = torch.zeros_like(sim_score) #refusal model
            logits = weight1[:, None, None]*logits1 + weight2[:, None, None]*logits2 + weight3[:, None, None]*logits3
            import pdb; pdb.set_trace();
            # outputs = model(**batch)
            # import pdb; pdb.set_trace();
            input_string, gen_output, gt = run_generation_two_adapters(cfg, batch, base_model, adapter1_model, adapter2_model, sim_score, tokenizer=tokenizer)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
              
          else:
            import pdb; pdb.set_trace();
            outputs = model(**batch)
            input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
            
        gt_loss = get_batch_loss(logits, batch['labels'])
        num_token_gt = (batch['labels']!=-100).sum(-1)
        gt_loss_per_token = gt_loss/num_token_gt

        if 'avg_gt_loss' not in eval_logs:
            eval_logs['avg_gt_loss'] = {}
        if 'gt_loss' not in eval_logs:
            eval_logs['gt_loss'] = {}
        if 'num_token_gt' not in eval_logs:
            eval_logs['num_token_gt'] = {}
        if 'generated_text' not in eval_logs:
            eval_logs['generated_text'] = {}
        # print(gt_loss.shape, num_token_gt.shape)
        eval_logs['avg_gt_loss'].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist())))
        eval_logs['gt_loss'].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist())))
        eval_logs['num_token_gt'].update(dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist())))
        eval_logs['generated_text'].update(dict(zip(indices.cpu().numpy().tolist(), zip(input_string, gen_output,gt))))


    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths, all_indices))
    eval_logs.update(eval_perturbation_ratio_two_adapters(base_eval_dataloader, perturb_dataloader, base_model, adapter1_model, adapter2_model, cfg))

    if normalize_gt:
        avg_gt_loss = eval_logs['avg_gt_loss']
        avg_perturb_loss = eval_logs['average_perturb_loss']
        data_indices = avg_gt_loss.keys()
        normalized_gt_loss = {}
        for idx in data_indices:
            truth_prob = np.exp(-1 * avg_gt_loss[idx])
            perturb_prob = np.exp(-1 * np.array(avg_perturb_loss[idx]))
            all_prob = np.array([truth_prob, *perturb_prob])
            normalized_gt_prob = truth_prob / all_prob.sum()
            normalized_gt_loss[idx] = -1 * np.log(normalized_gt_prob)

        eval_logs['normalized_gt_loss'] = normalized_gt_loss

    return eval_logs

@hydra.main(version_base=None, config_path="config", config_name="eval_everything")
def main(cfg):
    assert len(cfg.data_path)==len(cfg.split_list)==len(cfg.eval_task)==len(cfg.question_key)==len(cfg.answer_key)==len(cfg.base_answer_key)==len(cfg.perturbed_answer_key), "data_path, split, eval_task, question_key, and answer_key must be the same length"
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, token=HF_token)#model_id, token=HF_token)

    tokenizer.pad_token = tokenizer.eos_token
    max_length = 500
    batch_size = cfg.batch_size

    model = None
    device_map="auto"
    config = AutoConfig.from_pretrained(model_cfg["ft_model_path"], model=HF_token) #model_id, token=HF_token)
    for attempt in range(3):
        try:
        # do thing
            if cfg.use_pretrained:
                print(f"Loading pretrained from {model_id}")
                model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="false", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
            elif cfg.two_adapters:
                base_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="false", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
                base_model_copy = deepcopy(base_model)
                base_model_copy_2 = deepcopy(base_model)
                adapter1_model = PeftModel.from_pretrained(base_model_copy, cfg.adapters_name)
                adapter1_model = adapter1_model.merge_and_unload()
                adapter2_model = PeftModel.from_pretrained(base_model_copy_2, cfg.adapters_name_2)
                adapter2_model = adapter2_model.merge_and_unload()
            else:
                print(f"Loading checkpoint from {cfg.model_path}")
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="false", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
                # import pdb; pdb.set_trace();    
        except Exception as e:
            print(e)
            continue
        # perhaps reconnect, etc.
        else:
            break
    else:
        print("Error: could not load model")
    if cfg.two_adapters:
        base_model = base_model.eval()
        adapter1_model = adapter1_model.eval()
        adapter2_model = adapter2_model.eval()
    else:
        model = model.eval()
    
    def reinitialize_weights(model) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    if cfg.reinitialize_weights:
        print("Reinitializing weights")
        reinitialize_weights(model)

    #write custom eval loop using compute_metrics

    aggregated_eval_logs = {}
    for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(cfg.data_path, cfg.split_list, cfg.question_key, cfg.answer_key, cfg.eval_task, cfg.base_answer_key, cfg.perturbed_answer_key)):
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        print(f'Working on eval task {eval_task} with split {split}')
        save_filename = os.path.join(cfg.save_dir, f"{eval_task}.json")
        save_filename = save_filename if world_size == 1 else os.path.join(cfg.save_dir, f"{eval_task}_{os.environ.get('LOCAL_RANK', '0')}.json")

        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            continue

        eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)

        normalize_gt = False 
        if 'eval_log' not in eval_task:
            normalize_gt = True
        if cfg.two_adapters:

            # if cfg.split == split:
            #     cfg.weight1, cfg.weight2, cfg.weight3 = 0.5, 0.5, 0.0
            # else:
            #     cfg.weight1, cfg.weight2, cfg.weight3 = 0.0, 0.0, 1.0
            
            eval_logs = get_all_evals_two_adapters(cfg, base_model, adapter1_model, adapter2_model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)
        else:
            eval_logs = get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)

        with open(save_filename, "w") as f:
            # pretty write json to f
            json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

    aggregated_eval_log_filename = os.path.join(cfg.save_dir, "eval_log_aggregated.json")

    with open(aggregated_eval_log_filename, "w") as f:
        # pretty write json to f
        json.dump(aggregated_eval_logs, f, indent=4)
                    

def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def run_generation(cfg, batch, model, tokenizer):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    split_symbol = " [/INST]" if cfg.model_family == 'llama2-7b' or 'llama2-7b-chat' else 'Answer: '
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    #add ["/INST "] to the end of each string
    if cfg.model_family == 'llama2-7b' or 'llama2-7b-chat':
        input_strings = [s + split_symbol for s in input_strings]
        
    #we only want to retain the input before the [/INST] token. split each string to only retain the content before the [/INST] token
    # ground_truth = [s.split("[/INST] ")[1] for s in input_strings]
    # input_strings = [s.split("[/INST] ")[0] for s in input_strings]
    # #add ["/INST "] to the end of each string
    # input_strings = [s + "[/INST] " for s in input_strings]
    
    #now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id


    inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
    #now generate
    out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=cfg.generation.max_length, max_new_tokens=cfg.generation.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    # import pdb; pdb.set_trace();
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return input_strings, strs, ground_truth

def run_generation_two_adapters(cfg, batch, base_model, adapter1_model, adapter2_model, sim_score, tokenizer):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    split_symbol = " [/INST]" if cfg.model_family == 'llama2-7b' or 'llama2-7b-chat' else 'Answer: '
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    #add ["/INST "] to the end of each string
    if cfg.model_family == 'llama2-7b' or 'llama2-7b-chat':
        input_strings = [s + split_symbol for s in input_strings]
        
    #we only want to retain the input before the [/INST] token. split each string to only retain the content before the [/INST] token
    # ground_truth = [s.split("[/INST] ")[1] for s in input_strings]
    # input_strings = [s.split("[/INST] ")[0] for s in input_strings]
    # #add ["/INST "] to the end of each string
    # input_strings = [s + "[/INST] " for s in input_strings]
    
    #now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    if cfg.two_adapters:
        inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(base_model.device)
    else:
        inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
    #now generate
    orig_len = inputs.input_ids.shape[-1]
    # import pdb; pdb.set_trace()
    weight3 = (sim_score<cfg.sim_threshold)*1.0
    weight1 = (sim_score>=cfg.sim_threshold)*1.0
    weight2 = torch.zeros_like(sim_score)

    for i in range(cfg.generation.max_length-orig_len):
        logits1=adapter1_model(inputs.input_ids, attention_mask=inputs.attention_mask).logits
        logits2=adapter2_model(inputs.input_ids, attention_mask=inputs.attention_mask).logits
        logits3=base_model(inputs.input_ids, attention_mask=inputs.attention_mask).logits
        logits = weight1[:, None, None]*logits1 + weight2[:, None, None]*logits2 + weight3[:, None, None]*logits3
        # logits = cfg.weight1*logits1+cfg.weight2*logits2+cfg.weight3*logits3
        probabilities = F.softmax(logits, dim=-1)
        predicted_indices = torch.argmax(probabilities, dim=-1)
        # import pdb; pdb.set_trace();
        inputs.input_ids = torch.cat([inputs.input_ids.cuda(), predicted_indices[:, -1:].cuda()], -1)
        # import pdb; pdb.set_trace();
        inputs.attention_mask = torch.cat([inputs.attention_mask.cuda(), torch.ones_like(predicted_indices[:, -1:]).cuda()], -1)
    # print(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    
    strs = left_pad_tokenizer.batch_decode(inputs.input_ids[:, orig_len:], skip_special_tokens=True)
    # import pdb; pdb.set_trace()
    print(strs)


    # out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=cfg.generation.max_length, max_new_tokens=cfg.generation.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    # strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return input_strings, strs, ground_truth

def eval_bleu(gen_outputs, ground_truths):

    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)


    eval_result = {
        'rouge': rouge_res,
        'bleu': bleu_res,
    }
    return eval_result

def eval_rouge_recall(gen_outputs, ground_truths, indices):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores['rouge1'].recall
        rougeL_recall[idx] = rouge_scores['rougeL'].recall


    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}

if __name__ == "__main__":
    main()


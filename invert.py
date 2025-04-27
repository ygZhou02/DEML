import argparse
import torch
import os
import re
import sys
import time
import pickle
import copy
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering)
from accelerate import Accelerator, dispatch_model
from dataset import Dataset
from utils import *


def get_model(base_model_name,
              lora_model_name,
              model_kwargs={"low_cpu_mem_usage": True, "use_cache": False}):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        use_fast=False
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    device_map = {}
    model_layers = len(base_model.model.layers)
    if lora_model_name is None:
        for i in range(model_layers):
            layer = "model.layers." + str(i)
            device_map[layer] = int(i / (model_layers) * torch.cuda.device_count())
        device_map["model.embed_tokens"] = 0
        device_map["model.norm"] = torch.cuda.device_count() - 1
        device_map["lm_head"] = torch.cuda.device_count() - 1
        model = dispatch_model(base_model, device_map=device_map)
    else:
        for i in range(model_layers):
            layer = "base_model.model.model.layers." + str(i)
            device_map[layer] = int(i / (model_layers) * torch.cuda.device_count())
        device_map["base_model.model.model.embed_tokens"] = 0
        device_map["base_model.model.model.norm"] = torch.cuda.device_count() - 1
        device_map["base_model.model.lm_head"] = torch.cuda.device_count() - 1
        lora_model = PeftModel.from_pretrained(base_model, lora_model_name, torch_dtype=torch.float16)
        model = dispatch_model(lora_model, device_map=device_map)
    model.gradient_checkpointing = True
    return tokenizer, model


def get_hidden_state(tokenizer, model, layer_id, prompt=None, input_embed=None, target_attention_mask=None):
    assert(prompt != None or input_embed != None)
    hidden_state_list = []
    hook_handles = []
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)
    def full_hook(module, input, output):
        '''full hook'''
        if isinstance(input, tuple):
            hidden_state_list.append(input[0])
        else:
            hidden_state_list.append(input)
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)
    if prompt != None:
        with torch.no_grad():
            target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
            target_input_ids = target_token['input_ids'].to(model.device)
            target_attention_mask = target_token['attention_mask'].to(model.device)
            inputs = {'input_ids': target_input_ids, 'attention_mask': target_attention_mask}
            
            '''get hidden states from all layers'''
            for name, module in model.named_modules():
                if len(name) > 0 and name[-1].isdigit():
                    if not name[-2].isdigit() and int(name[-1]) == 0:
                        handle = module.register_forward_hook(full_hook)
                        hook_handles.append(handle)
                    elif not name[-2].isdigit() and int(name[-1]) <= layer_id:
                        handle = module.register_forward_hook(forward_hook)
                        hook_handles.append(handle)
                    elif name[-2].isdigit() and int(name[-2:]) <= layer_id:
                        handle = module.register_forward_hook(forward_hook)
                        hook_handles.append(handle)
                    # elif name.endswith("model.norm"):
                    #     handle = module.register_forward_hook(forward_hook)
                    #     hook_handles.append(handle)

            next_ = model(**inputs)
            embed_layer = model.model.get_input_embeddings()
            ori_input_embed = embed_layer(target_input_ids)

    elif input_embed != None:
        with torch.no_grad():
            input_embed.to(model.device)
            new_inputs = {'inputs_embeds': input_embed, 'attention_mask': target_attention_mask}

            '''get hidden states from all layers'''
            for name, module in model.named_modules():
                if name[-1].isdigit():
                    if not name[-2].isdigit() and int(name[-1]) == 0:
                        handle = module.register_forward_hook(full_hook)
                        hook_handles.append(handle)
                    elif not name[-2].isdigit() and int(name[-1]) <= layer_id:
                        handle = module.register_forward_hook(forward_hook)
                        hook_handles.append(handle)
                    elif name[-2].isdigit() and int(name[-2:]) <= layer_id:
                        handle = module.register_forward_hook(forward_hook)
                        hook_handles.append(handle)
                    # elif name.endswith("model.norm"):
                    #     handle = module.register_forward_hook(forward_hook)
                    #     hook_handles.append(handle)

            next_ = model(**new_inputs)
            ori_input_embed = input_embed
            target_input_ids = None
    else:    
        raise NotImplementedError
    for handle in hook_handles:
        handle.remove()
    return target_input_ids, target_attention_mask, ori_input_embed, hidden_state_list


def get_hidden_state_base(tokenizer, model, layer_id, prompt=None, input_embed=None, target_attention_mask=None):
    assert(prompt != None or input_embed != None)
    hidden_state_list = []
    hook_handles = []
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)
    def full_hook(module, input, output):
        if isinstance(input, tuple):
            hidden_state_list.append(input[0])
        else:
            hidden_state_list.append(input)
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)
    if prompt != None:
        with torch.no_grad():
            target_token = tokenizer(prompt, padding=True, truncation=False, return_tensors='pt')
            target_input_ids = target_token['input_ids'].to(model.device)
            target_attention_mask = target_token['attention_mask'].to(model.device)
            inputs = {'input_ids': target_input_ids, 'attention_mask': target_attention_mask}
            
            '''get hidden states from all layers'''
            for name, module in model.named_modules():
                if name[-1].isdigit():
                    if not name[-2].isdigit() and int(name[-1]) == 0:
                        handle = module.register_forward_hook(full_hook)
                        hook_handles.append(handle)
                    elif not name[-2].isdigit() and int(name[-1]) <= layer_id:
                        handle = module.register_forward_hook(forward_hook)
                        hook_handles.append(handle)
                    elif name[-2].isdigit() and int(name[-2:]) <= layer_id:
                        handle = module.register_forward_hook(forward_hook)
                        hook_handles.append(handle)
                    # elif name.endswith("model.norm"):
                    #     handle = module.register_forward_hook(forward_hook)
                    #     hook_handles.append(handle)

            next_ = model(**inputs)
            embed_layer = model.model.get_input_embeddings()
            ori_input_embed = embed_layer(target_input_ids)

    elif input_embed != None:
        with torch.no_grad():
            input_embed.to(model.device)
            new_inputs = {'inputs_embeds': input_embed, 'attention_mask': target_attention_mask} 
            
            '''get hidden states from all layers'''
            for name, module in model.named_modules():
                if name[-1].isdigit():
                    if not name[-2].isdigit() and int(name[-1]) == 0:
                        handle = module.register_forward_hook(full_hook)
                        hook_handles.append(handle)
                    elif not name[-2].isdigit() and int(name[-1]) <= layer_id:
                        handle = module.register_forward_hook(forward_hook)
                        hook_handles.append(handle)
                    elif name[-2].isdigit() and int(name[-2:]) <= layer_id:
                        handle = module.register_forward_hook(forward_hook)
                        hook_handles.append(handle)
                    # elif name.endswith("model.norm"):
                    #     handle = module.register_forward_hook(forward_hook)
                    #     hook_handles.append(handle)

            next_ = model(**new_inputs)
            ori_input_embed = input_embed
            target_input_ids = None
    else:    
        raise NotImplementedError
    for handle in hook_handles:
        handle.remove()
    return target_input_ids, target_attention_mask, ori_input_embed, hidden_state_list


def invert_embedding(hidden_state, tokenizer, embed_layer, total_input_ids, f=None, invert_method='cosine', show_position=False, filter_nonascii=True, top_k=10):
    if f != None:
        sys.stdout = f
    if len(hidden_state.shape) >= 3:
        new_input_embed_squeeze = hidden_state.squeeze(0)
    else:
        new_input_embed_squeeze = hidden_state

    ret_list = []
    for embed in new_input_embed_squeeze:
        if invert_method == 'L2':
            '''show by L2 distance'''
            dist_ret = -torch.norm(embed_layer.weight - embed, p=2, dim=1)
        elif invert_method == 'cosine':
            '''show by cosine similarity'''
            dist_ret = F.cosine_similarity(embed.type(torch.float32), embed_layer.weight.type(torch.float32), dim=-1).detach().cpu()
        else:
            raise NotImplementedError
        idx = 0
        if filter_nonascii:
            p = torch.topk(dist_ret, top_k).indices[idx]
            while not tokenizer.decode([p]).isascii() or p == 0 or p == 2:
                idx += 1
                if idx == top_k:
                    idx = 0
                    print("fail to filter non ascii"); break
                p = torch.topk(dist_ret, top_k).indices[idx]
        ret_list.append(torch.topk(dist_ret, top_k).indices[idx])
        
    '''show position accuracy'''
    acc_cnt = 0
    prompt_length = len(total_input_ids[0])
    for j in range(prompt_length):
        if total_input_ids[0][j] == ret_list[j]:
            acc_cnt += 1
    acc = acc_cnt / (len(ret_list))
    ret_tokens = tokenizer.decode(torch.tensor(ret_list[1:]))
    if f != None:
        sys.stdout = sys.__stdout__
    return acc, ret_tokens, ret_list


def invert_and_find_best(hidden_state, gt_hidden_state, tokenizer, model, 
                         total_input_ids, layer_id, f=None, invert_method='cosine', 
                         filter_nonascii=True, add_perplexity=True, top_k_ppl=10, top_k_cos=10):
    if f != None:
        sys.stdout = f
    embed_layer = model.model.get_input_embeddings()
    if len(hidden_state.shape) >= 3:
        new_input_embed_squeeze = hidden_state.squeeze(0)
    else:
        new_input_embed_squeeze = hidden_state

    ret_list = []
    ret_top_k = []
    if top_k_cos == 0:
        for embed in new_input_embed_squeeze:
            ret_top_k.append([0])
            ret_list.append(0)
    else:
        for embed in new_input_embed_squeeze:
            if invert_method == 'L2':
                '''show by L2 distance'''
                dist_ret = -torch.norm(embed_layer.weight - embed, p=2, dim=1)
            elif invert_method == 'cosine':
                '''show by cosine similarity'''
                dist_ret = F.cosine_similarity(embed.type(torch.float32), embed_layer.weight.type(torch.float32), dim=-1).detach().cpu()
            else:
                raise NotImplementedError
            ret_top_k.append(torch.topk(dist_ret, top_k_cos).indices.tolist())
            idx = 0
            if filter_nonascii:
                p = torch.topk(dist_ret, top_k_cos).indices[idx]
                while not tokenizer.decode([p]).isascii() or p == 0 or p == 2:
                    idx += 1
                    if idx == top_k_cos:
                        idx = 0
                        print("fail to filter non ascii"); break
                    p = torch.topk(dist_ret, top_k_cos).indices[idx]
            ret_list.append(int(torch.topk(dist_ret, top_k_cos).indices[idx].data.cpu()))

    print("......start replacing......")
    start = time.time()
    for i, top_list in enumerate(ret_top_k):
        if i > 0 and add_perplexity:
            input_ids = copy.deepcopy(ret_list[:i])
            perplexity, topk_ids = get_perplexity(input_ids, model, layer_id=layer_id, top_k=top_k_ppl)
            top_list += topk_ids.tolist()
        
        replaced_ret_list = []
        for item in top_list:
            replaced_ret = copy.deepcopy(ret_list)
            replaced_ret[i] = item
            replaced_ret_list.append(replaced_ret)

        new_hidden_states = forward_and_get_last_hidden_state(model, replaced_ret_list, None, layer_id=layer_id)
        gt_hidden_state = gt_hidden_state.to(new_hidden_states.device)
        cos_sim_list = F.cosine_similarity(new_hidden_states.index_select(1, torch.tensor([i]).to(new_hidden_states.device)).permute(1,0,2).squeeze(0).type(torch.float32), 
                                           gt_hidden_state.index_select(1, torch.tensor([i]).to(new_hidden_states.device)).permute(1,0,2).squeeze(0).type(torch.float32), dim=-1)
        cos_sim_list = cos_sim_list.data.cpu().numpy()
        idx = np.argmax(cos_sim_list)
        ret_list[i] = top_list[idx]

    end = time.time()
    print("replace cost: {} s".format(end-start))
    print("......calculating accuracy......")
    '''show accuracy'''
    acc_cnt = 0
    prompt_length = len(total_input_ids[0])
    for j in range(prompt_length):
        if total_input_ids[0][j] == ret_list[j]:
            acc_cnt += 1
    acc = acc_cnt / (len(ret_list))
    print("acc: ", acc)
    ret_tokens = tokenizer.decode(torch.tensor(ret_list[1:]))
    if f != None:
        sys.stdout = sys.__stdout__
    return acc, ret_tokens, ret_list


def forward_and_get_last_hidden_state(model, input_ids, attention_mask, layer_id):
    if len(torch.tensor(input_ids).shape) < 2:
        input_ids_squeeze = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    else:
        input_ids_squeeze = torch.tensor(input_ids).to(model.device)
    new_inputs = {'input_ids': input_ids_squeeze, 'attention_mask': attention_mask} 

    hidden_state_list = []
    hook_handles = []
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)

    for name, module in model.named_modules():
        if len(name) > 1 and name[-2:] == str(layer_id):
            handle = module.register_forward_hook(forward_hook)
            hook_handles.append(handle)
    phi_relaxed = model(**new_inputs)
    for handle in hook_handles:
        handle.remove()
    last_hidden_state = hidden_state_list[0]
    return last_hidden_state


def get_perplexity(input_ids, model, layer_id, next_ids=None, top_k=None):
    hidden_state_list = []
    hook_handles = []
    if isinstance(input_ids, torch.Tensor):
        inputs = {'input_ids': input_ids.to(model.device), 'attention_mask': None}
    else:
        inputs = {'input_ids': torch.tensor(input_ids).unsqueeze(0).to(model.device), 'attention_mask': None}
    def forward_hook(module, input, output):
        if isinstance(output, tuple):
            for item in output:
                hidden_state_list.append(item)
        else:
            hidden_state_list.append(output)

    for name, module in model.named_modules():
        if len(name) > 1 and name[-2:] == str(layer_id):
            handle = module.register_forward_hook(forward_hook)
            hook_handles.append(handle)
    next_token_logits = model(**inputs).logits[:, -1, :]
    filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)
    probs = F.softmax(filtered_next_token_logits, dim=-1)
    for handle in hook_handles:
        handle.remove()
    if next_ids != None:
        perplexity = probs[0][next_ids]
        top_ids = next_ids
    elif top_k != None:
        top_ids = torch.topk(probs[0], top_k).indices
        perplexity = probs[0][top_ids]
    else:
        raise NotImplementedError
    return perplexity, top_ids


def main(args):
    '''create log file'''
    lora_setting = "with_lora" if args.lora_model_name is not None else "no_lora"
    txt_file = open("{}-{}-{}-{}-{}-{}-{}-{}.log".format(lora_setting, *time.localtime()), "w")
    txt_file.write("invert settings: \nbase model: {}\nloramodel: {}\n Dataset: {}\n".format(args.base_model_name, args.lora_model_name, args.dataset_path))
    txt_file.write(str(vars(args)))
    np.random.seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    '''load prompt dataset'''
    prompt_dataset = Dataset(dataset_name=args.dataset_path, dataset_type=args.dataset_type)
    prompt_dataset.data = prompt_dataset.data[:args.dataset_len]

    '''get model'''
    tokenizer, model = get_model(base_model_name=args.base_model_name, lora_model_name=args.lora_model_name)

    '''freeze model parameter'''
    for param in model.parameters():
        param.requires_grad = False

    '''fix <start> token'''
    embed_layer = model.model.get_input_embeddings()
    START_EMBED = embed_layer.weight[0].data
    START_EMBED = START_EMBED.unsqueeze(0).unsqueeze(0)
    START_EMBED.requires_grad_(False)

    '''get range'''
    embed_matrix = np.array(embed_layer.weight.data.cpu())
    left_range = get_sorted_top_k(embed_matrix, top_k=10, axis=0, reverse=False)
    left_range = torch.FloatTensor(left_range[0][-1]).type(torch.float16).to(model.device)
    right_range = get_sorted_top_k(embed_matrix, top_k=10, axis=0, reverse=True)
    right_range = torch.FloatTensor(right_range[0][-1]).type(torch.float16).to(model.device)

    input_ids_list = []
    attention_mask_list = []
    all_hidden_states_list = []

    for prompt_ in prompt_dataset.get_data():
        '''get all hidden states in a list'''
        total_input_ids, total_attention_mask, _, all_hidden_states = get_hidden_state(tokenizer, 
                    model, layer_id=args.num_invert_layers, prompt=prompt_)
        txt_file.write("collected hidden states: {} \n".format(len(all_hidden_states)))
        input_ids_list.append(total_input_ids)
        attention_mask_list.append(total_attention_mask)
        all_hidden_states_list.append(all_hidden_states)

    if args.lora_model_name is not None:
        '''disable lora'''
        # model = model.merge_and_unload(progressbar=True)
        # model.unmerge_adapter()
        # model.merge_adapter()
        model.unload()

    for i, prompt in enumerate(prompt_dataset.get_data()):
        txt_file.write("recovering {}\n".format(prompt))
        total_input_ids, total_attention_mask, all_hidden_states = input_ids_list[i], attention_mask_list[i], all_hidden_states_list[i]

        if args.lora_model_name is not None:
            total_input_ids_nolora, total_attention_mask_nolora, _, all_hidden_states_nolora = get_hidden_state_base(tokenizer, 
                        model, layer_id=args.num_invert_layers, prompt=prompt)
            txt_file.write("collected hidden states: {} \n".format(len(all_hidden_states_nolora)))
            txt_file.write("cos sim: {}\n".format(F.cosine_similarity(all_hidden_states_nolora[-1].type(torch.float32), all_hidden_states[-1].type(torch.float32), dim=-1)))
            txt_file.write("L2: {}\n".format(torch.norm(all_hidden_states_nolora[-1].type(torch.float32) - all_hidden_states[-1].type(torch.float32), p=2, dim=-1).detach().cpu()))

        prompt_length = len(total_input_ids[0])
        recover_length = prompt_length
        target_input_ids = total_input_ids
        target_attention_mask = total_attention_mask
        next_hidden_states_last = all_hidden_states[-1].detach().requires_grad_(False)
        txt_file.write("recovering piece length: {}\n".format(prompt_length))

        '''define hyper-params'''
        loss_func = torch.nn.MSELoss(reduction='mean')
        lr = args.lr
        total_epoch = args.epoch
        alpha = args.alpha
        
        '''init input embed'''
        size = (len(target_input_ids[0]) - 1, START_EMBED.shape[-1])
        if args.init_method == "gaussian":
            means = torch.zeros(size)
            new_input_embed_0 = torch.normal(mean=means, std=args.init_param)
            new_input_embed_0 = new_input_embed_0.unsqueeze(0).type(torch.float16)
        elif args.init_method == "uniform":
            new_input_embed_np = np.random.uniform(low=-args.init_param, high=args.init_param, size=size)
            new_input_embed_0 = torch.FloatTensor(new_input_embed_np)
        else:
            raise NotImplementedError
            
        new_input_embed_0 = new_input_embed_0.unsqueeze(dim=0).type(torch.float16).to(model.device)
        new_input_embed_0.requires_grad_(True)

        epochs = []
        loss_lst = []
        cos_sim_lst = []

        '''weighted average loss'''
        weight_mask = init_weight_mask(0, recover_length, method="linear", devices=[model.device])
        part_epoch = total_epoch
        
        '''start timer'''
        start = time.time()

        for i in range(part_epoch):
            '''clip embedding'''
            if args.clip:
                with torch.no_grad():
                    clip_range = 0.2
                    new_input_embed_0 = torch.clip(new_input_embed_0, -clip_range, clip_range)
            new_input_embed_0 = new_input_embed_0.requires_grad_(True)
            optim = torch.optim.SGD([new_input_embed_0], lr=lr)

            '''add start token'''
            new_input_embed_ = torch.cat((START_EMBED, new_input_embed_0), dim=1)

            '''||phi(relaxed(Z, T)) - phi(x*)||**2'''
            new_inputs = {'inputs_embeds': new_input_embed_, 'attention_mask': target_attention_mask} 

            hidden_state_list = []
            hook_handles = []
            def forward_hook(module, input, output):
                if isinstance(output, tuple):
                    for item in output:
                        hidden_state_list.append(item)
                else:
                    hidden_state_list.append(output)
    
            '''get hidden states from all layers'''
            for name, module in model.named_modules():
                if len(name) > 1 and name[-2:] == str(args.num_invert_layers):
                    handle = module.register_forward_hook(forward_hook)
                    hook_handles.append(handle)
            phi_relaxed = model(**new_inputs)
            for handle in hook_handles:
                handle.remove()
            last_hidden_state = hidden_state_list[0]
            hidden_state_list = []

            '''compute mse loss'''
            next_hidden_states_last = next_hidden_states_last.to(last_hidden_state.device)
            loss_mse = loss_func(last_hidden_state.type(torch.float32), next_hidden_states_last.type(torch.float32))
            # print("{} epoch, {} loss".format(i, loss_mse.data))

            '''compute similarity'''
            cos_sim = F.cosine_similarity(last_hidden_state.type(torch.float32), next_hidden_states_last.type(torch.float32), dim=-1)

            '''backward'''
            optim.zero_grad()
            cos_sim = cos_sim.to(model.device)
            sum_cos_sim = ((-cos_sim) * weight_mask).sum()

            relu_loss = F.relu(torch.abs(new_input_embed_) - right_range).sum()
            loss = sum_cos_sim + alpha * relu_loss 
            if args.optim_method == "MSELoss":
                loss_mse.backward(inputs=[new_input_embed_0])    # optimize by MSELoss
            elif args.optim_method == "cosine":
                loss.backward(inputs=[new_input_embed_0])  # optimize by cosine sim
            
            if torch.any(torch.isnan(cos_sim)):
                txt_file.write("encounter NAN\n")
                break
            optim.step()
            epochs.append(i)                              # for loss graph
            loss_lst.append(relu_loss.data.cpu())         # for loss graph
            cos_sim_lst.append(sum_cos_sim.data.cpu())    # for loss graph
            torch.cuda.empty_cache()

            if (i+1) % 25 == 0 or i == part_epoch - 1:
                txt_file.write("{} epoch, cos sim: {}\n".format(i, cos_sim.mean().data))
                txt_file.write("relu loss: {}\n".format(relu_loss.data))
                txt_file.write("total loss: {}\n".format(loss))

            if (i+1) % 100 == 0 or i == part_epoch - 1:
                end = time.time()
                acc, ret_tokens, ret_list = invert_embedding(torch.cat((START_EMBED, new_input_embed_0), dim=1), 
                                                            tokenizer, 
                                                            embed_layer, 
                                                            total_input_ids, 
                                                            invert_method=args.invert_method,
                                                            filter_nonascii=args.filter_nonascii)
                txt_file.write("0 layer hidden state ret: lr{}, epoch{}, acc{}, cos sim{}, final loss{}, time {}s, alpha {}, result token: \n{}\n".format(lr, i, acc, cos_sim.mean(), relu_loss, end-start, alpha, ret_tokens))
                
        '''best inversion policy'''
        acc, ret_tokens, ret_list = invert_and_find_best(torch.cat((START_EMBED, new_input_embed_0), dim=1), 
                                                        next_hidden_states_last,  
                                                        tokenizer, 
                                                        model, 
                                                        total_input_ids,
                                                        layer_id=args.num_invert_layers,
                                                        f=txt_file, 
                                                        invert_method=args.invert_method,
                                                        filter_nonascii=args.filter_nonascii,
                                                        add_perplexity=args.perplexity,
                                                        top_k_ppl=args.top_k_ppl,
                                                        top_k_cos=args.top_k_cos)
        txt_file.write("acc : {},\n replaced token :\n{}\n".format(acc, ret_tokens))

    txt_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="medalpaca/medical_meadow_wikidoc")  # "data/airline.json"
    parser.add_argument("--dataset-type", type=str, default="datasets",
                        choices=["local", "datasets", "github"])
    parser.add_argument("--dataset-len", type=int, default=100)
    parser.add_argument("--base-model-name", type=str, required=True)
    parser.add_argument("--lora-model-name", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--clip", type=bool, default=True)
    parser.add_argument("--num-invert-layers", type=int, default=30, 
                        choices=range(1, 80))
    parser.add_argument("--init-method", type=str, default="uniform")
    parser.add_argument("--init-param", type=float, default=0.1)
    parser.add_argument("--optim-method", type=str, default="cosine",
                        choices=["cosine", "MSELoss"])
    parser.add_argument("--invert-method", type=str, default="cosine",
                        choices=["cosine", "L2"])
    parser.add_argument("--show-low-confidence", type=bool, default=False)
    parser.add_argument("--fine-tune", type=bool, default=False)
    parser.add_argument("--filter-nonascii", type=bool, default=True)
    parser.add_argument("--perplexity", type=bool, default=True)
    parser.add_argument("--top-k-ppl", type=int, default=10)
    parser.add_argument("--top-k-cos", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="results") 
    
    args = parser.parse_args()
    main(args)
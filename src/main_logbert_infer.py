from csv_utils import *
from models_sup_blend import *
from time import strftime, time
import random
import logging
import re
import os
import torch.nn.functional as F
import sys
from itertools import product, chain, combinations
from collections import defaultdict
import argparse
from tqdm import tqdm


cfg = {
    "blend_warmup": True,
    "train_blend_warmup": False,
    "n_epochs_blend_only": 5,

    "debug_mode": False, #True,

    "device": None,
    "pt_model": "bert-base-uncased",
    "n_trials": 1,
    "n_epochs": 50,
    "learn_rate": 1e-5,
    "epsilon": 1e-8,
    "pivot_metric": "auc",
    "early_stop": 3,
    "tasks": {
        "main": {
            "classes": None, 
        },
        "nli": {
            "data_paths": ["../data/rsc/mnli-org.csv",
                           "../data/rsc/antsyn-nli.csv"],
            "classes": ["ent", "con", "neu"],
        },
        "senti": {
            "data_paths": ["../data/rsc/senti-irish.csv",
                           "../data/rsc/senti-ldong.csv",
                           "../data/rsc/senti-mm.csv",
                           "../data/rsc/senti-sem17.csv",
                           "../data/rsc/senti-norm.csv"],
            "classes": ["pos", "neg", "neu"],
        },
        "causal": {
            "data_paths": ["../data/rsc/because-causal.csv",
                           "../data/rsc/conet-causal.csv",
                           "../data/rsc/pdtb-i-causal.csv",
                           "../data/rsc/wiqa-causal.csv"],
            "classes": ["cause", "obstruct", "precede", "sync", "else"],
        },
        "normarg_polar": {
            "data_path": "../data/rsc/normarg.csv",
            "classes": ["consist", "contrast"],
        },
        "normarg_jtype": {
            "data_path": "../data/rsc/normarg.csv",
            "classes": ["norm", "conseq"],
        },
        "normarg_senti": {
            "data_path": "../data/rsc/normarg.csv",
            "classes": ["positive", "negative"],
            "con_classes": ["advocate", "object"],
        },
 
    },
    "max_n_batch_tokens": 512*5,
    "max_batch_size": 6,
    "data_dir": None,
    "logs_dir": None,
    "save_model": False,
    "models_dir": "../models",
    "rel2label": {
            "1": 0,  # support
            "-1": 1,  # attack
            "0": 2  # neutral
    }
}

def load_data_main(trainer, data_idx):
    n_max_insts = 50000
    insts = []
    for r, row in tqdm(enumerate(iter_csv_header(cfg["data_path"]))):
        if r < data_idx: continue

        insts.append({
            "id": row["pairid"],
            "pre_text": row["text_from"],
            "con_text": row["text_to"],
        })

        if len(insts) == n_max_insts: break

    batches = trainer.make_batches(insts, ["pre", "con"])
    return batches, len(insts)


blend_methods = {
    "no": [0] * 100,
}

def iter_comb(items):
    for n in range(1, len(items)+1):
        for subitems in combinations(items, n):
            yield list(subitems)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-logic", nargs="*", 
                        choices=["nli", "senti", "causal", "normarg"],
                        default=["nli", "senti", "causal", "normarg"],
                        help="Logic tasks to run.")
    parser.add_argument("-data", required=True)
    parser.add_argument("-out", required=True)
    parser.add_argument("-model", required=True)
    parser.add_argument("-cuda", type=int, required=True, 
                        help="CUDA device to use.")

    args = parser.parse_args()

    cfg["data_path"] = args.data
    cfg["device"] = f'cuda:{args.cuda}'
    cfg["tasks"]["main"]["classes"] = ["sup", "att", "neu"]
    cfg["logs_dir"] = f'../logs/'
    cfg["blend_warmup"] = True
    cfg["blend_type"] = "no"

    os.makedirs(cfg["logs_dir"], exist_ok=True)
    tasks = ["main"] + args.logic

    model_path = args.model
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    print("\n=============================================================")
    print(tasks, cfg["blend_type"])
    print("==============================================================")

    cfg["blend_rate"] = blend_methods[cfg["blend_type"]]
    cfg["task2classes"] = {}
    for task in tasks:
        if task == "normarg":
            cfg["task2classes"]["normarg_polar"] = \
                            cfg["tasks"]["normarg_polar"]["classes"]
            cfg["task2classes"]["normarg_jtype"] = \
                            cfg["tasks"]["normarg_jtype"]["classes"]
            cfg["task2classes"]["normarg_norm_senti"] = \
                            cfg["tasks"]["normarg_senti"]["classes"]
            cfg["task2classes"]["normarg_conseq_senti"] = \
                            cfg["tasks"]["normarg_senti"]["classes"]
        else:
            cfg["task2classes"][task] = cfg["tasks"][task]["classes"]


    # Prepare the trainer
    trainer = Trainer(cfg)

    # Warmup with blend tasks
    init_model = torch.load(model_path)
    trainer.init_model(init_model)
    trainer.model = trainer.model.to(cfg["device"])

    # Prediction
    print("Predict...")
    with open(args.out, "w") as f:
        keys = ["id", "con_text", "pre_text", "label_prob", "label_pred"]
        out_csv = csv.writer(f)
        out_csv.writerow(keys)

        data_idx = 0
        while True:
            print(f"Loading data from index {data_idx}...")
            batches, n_insts = load_data_main(trainer, data_idx)
            print("N={}".format(sum(len(b["id"]) for b in batches)))
            if n_insts > 0: 
                data_idx += n_insts
            else:
                break

            trainer.run_epoch(batches, "predict", "main")

            for batch in batches:
                for i in range(len(batch["id"])):
                    out_csv.writerow([batch[key][i] for key in keys])
            


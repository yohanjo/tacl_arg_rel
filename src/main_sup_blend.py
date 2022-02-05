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
}

def load_data_main(trainer):
    pair2split = {}
    for mode in cfg["modes"]:
        for row in iter_csv_header(f'{cfg["data_dir"]}/' + \
                                   f'{cfg["data_prefix"]}-{mode}-split.csv'):
            pair2split[row["pairid"]] = row["split"]

    rel2label = {"1": 0, "-1": 1, "0": 2}
    split2insts = defaultdict(list)
    for mode in cfg["modes"]: 
        for row in iter_csv_header(f'{cfg["data_dir"]}/' + \
                                   f'{cfg["data_prefix"]}-{mode}.csv'):
            if row["pairid"] not in pair2split: continue
            split = pair2split[row["pairid"]]
            split2insts[split].append({
                "id": row["pairid"],
                "pre_text": row["text_from"],
                "con_text": row["text_to"],
                "label": rel2label[row["relation"]],
            })

    split2batches = {}
    for split, insts in split2insts.items():
        random.shuffle(insts)
        split2batches[split] = trainer.make_batches(insts, ["pre", "con"])

    return split2batches

def load_data_nli(trainer):
    cls2idx = {cls: idx for idx, cls in \
                    enumerate(cfg["tasks"]["nli"]["classes"])}
    insts = []
    for data_path in cfg["tasks"]["nli"]["data_paths"]:
        for row in iter_csv_header(data_path):
            if row["label"] not in cls2idx: continue

            label = cls2idx[row["label"]]
            inst = {
                "id": row["id"],
                "text1_text": row["text1"],
                "text2_text": row["text2"],
                "rel": row["label"],
                "label": label,
            }
            insts.append(inst)
            if cfg["debug_mode"] and  len(insts) > 100: break

    random.shuffle(insts)
    batches = trainer.make_batches(insts, ["text1", "text2"])

    return batches

def load_data_senti(trainer):
    cls2idx = {cls: idx for idx, cls in \
                    enumerate(cfg["tasks"]["senti"]["classes"])}
    insts = []
    for data_path in cfg["tasks"]["senti"]["data_paths"]:
        for row in iter_csv_header(data_path):
            if row["label"] not in cls2idx: continue

            label = cls2idx[row["label"]]
            inst = {
                "id": row["id"],
                "input_text": row["text"],
                "target_text": row["target"],
                "senti": row["label"],
                "label": label,
            }
            insts.append(inst)
            if cfg["debug_mode"] and  len(insts) > 100: break

    random.shuffle(insts)
    batches = trainer.make_batches(insts, ["input", "target"])

    return batches

def load_data_causal(trainer):
    cls2idx = {cls: idx for idx, cls in \
                    enumerate(cfg["tasks"]["causal"]["classes"])}
    insts = []
    for data_path in cfg["tasks"]["causal"]["data_paths"]:
        for row in iter_csv_header(data_path):
            if row["label"] not in cls2idx: continue

            label = cls2idx[row["label"]]
            inst = {
                "id": row["id"],
                "text1_text": row["text1"],
                "text2_text": row["text2"],
                "rel": row["label"],
                "label": label,
            }
            insts.append(inst)
            if cfg["debug_mode"] and  len(insts) > 100: break

    random.shuffle(insts)
    batches = trainer.make_batches(insts, ["text1", "text2"])

    return batches

def load_data_normarg_polar(trainer):
    cls2idx = {cls: idx for idx, cls in \
                    enumerate(cfg["tasks"]["normarg_polar"]["classes"])}
    insts = []
    data_path = cfg["tasks"]["normarg_polar"]["data_path"]
    for row in iter_csv_header(data_path):
        if row["conpre_polar"] not in cls2idx: continue

        label = cls2idx[row["conpre_polar"]]
        inst = {
            "id": row["id"],
            "pre_text": row["pre"],
            "con_text": row["con"],
            "polar": row["conpre_polar"],
            "label": label,
        }
        insts.append(inst)
        if cfg["debug_mode"] and  len(insts) > 100: break

    random.shuffle(insts)
    batches = trainer.make_batches(insts, ["pre", "con"])
    return batches

def load_data_normarg_jtype(trainer):
    cls2idx = {cls: idx for idx, cls in \
                    enumerate(cfg["tasks"]["normarg_jtype"]["classes"])}

    insts = {}
    data_path = cfg["tasks"]["normarg_jtype"]["data_path"]
    for row in iter_csv_header(data_path):
        if row["pre_jtype"] == "property":
            row["pre_jtype"] = "conseq"
        if row["pre_jtype"] not in cls2idx: continue

        conid, preid = row["id"].split("_")

        label = cls2idx[row["pre_jtype"]]
        insts[preid] = {
            "id": preid,
            "input_text": row["pre"],
            "jtype": row["pre_jtype"],
            "label": label,
        }
        if cfg["debug_mode"] and len(insts) > 100: break

    for row in iter_csv_header(data_path):
        conid, preid = row["id"].split("_")

        label = cls2idx["norm"]
        insts[conid] = {
            "id": conid,
            "input_text": row["con"],
            "jtype": "norm",
            "label": label,
        }
    insts = list(insts.values())

    random.shuffle(insts)
    batches = trainer.make_batches(insts, ["input"])
    return batches

def load_data_normarg_senti(trainer, jtypes_inc):
    cls2idx = {cls: idx for idx, cls in \
                    enumerate(cfg["tasks"]["normarg_senti"]["classes"])}

    insts = {}
    data_path = cfg["tasks"]["normarg_senti"]["data_path"]
    for row in iter_csv_header(data_path):
        if row["pre_jtype"] not in jtypes_inc: continue
        if row["pre_senti"] not in cls2idx: continue

        conid, preid = row["id"].split("_")

        label = cls2idx[row["pre_senti"]]
        insts[preid] = {
            "id": preid,
            "input_text": row["pre"],
            "senti": row["pre_senti"],
            "label": label,
        }
        if cfg["debug_mode"] and len(insts) > 100: break

    if "norm" in jtypes_inc:
        cls2idx = {cls: idx for idx, cls in \
                        enumerate(cfg["tasks"]["normarg_senti"]["con_classes"])}
        for row in iter_csv_header(data_path):
            conid, preid = row["id"].split("_")

            label = cls2idx[row["con_polar"]]
            insts[conid] = {
                "id": conid,
                "input_text": row["con"],
                "senti": row["con_polar"],
                "label": label,
            }
    insts = list(insts.values())

    random.shuffle(insts)
    batches = trainer.make_batches(insts, ["input"])
    return batches


def get_prefix():
    prefix = f'sup_{cfg["name"]}'
    prefix += "-{}".format(strftime("%Y%m%d_%H%M%S", localtime()))
    prefix += f'-DT{cfg["dataset"]}'
    prefix += f'-TY{cfg["dtype"]}'
    prefix += f'-BWup' if cfg["blend_warmup"] else "-BWno"
    prefix += f'-TS{"+".join(sorted(cfg["task2classes"].keys()))}'
    prefix += f'-BT{cfg["blend_type"]}'

    return prefix

def get_logger(path):
    logger = logging.getLogger(str(time()))
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    if path:
        logger.addHandler(logging.FileHandler(path, mode="w"))
    return logger
 
blend_methods = {
    "no": [0] * 100,
}

def iter_comb(items):
    for n in range(1, len(items)+1):
        for subitems in combinations(items, n):
            yield list(subitems)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", required=True,
                        choices=["basic", "logbert"],
                        help="Task (basic=vanilla BERT, logbert=LogBERT)")
    parser.add_argument("-dataset", choices=["kialo", "debate"], required=True,
                        help="Dataset.")
    parser.add_argument("-dtype", choices=["normative", "causal", "all"], 
                        required=True, 
                        help="Argument type (causal=non-normative).")
    parser.add_argument("-logic", nargs="*", 
                        choices=["nli", "senti", "causal", "normarg"],
                        help="Logic tasks to run.")
    parser.add_argument("-cuda", type=int, required=True, 
                        help="CUDA device to use.")
    parser.add_argument("-n_trials", type=int, default=1, 
                        help="Number of independent runs.")
    parser.add_argument("-save_model", action="store_true",
                        help="Save the trained model to disk.")
    parser.add_argument("-train_logic_tasks", action="store_true",
                        help="Train on logic tasks and save the result model. " + \
                             "Otherwise, it's assumed that a saved model " + \
                             "already exists.")

    args = parser.parse_args()

    cfg["data_prefix"] = f'{args.dataset}_{args.dtype}_pairs'
    cfg["data_dir"] = f'../data/{args.dataset}'
    cfg["device"] = f'cuda:{args.cuda}'
    cfg["name"] = "blend"
    cfg["dataset"] = args.dataset
    cfg["dtype"] = args.dtype
    cfg["logs_dir"] = f'../logs/sup-{args.dataset}'
    cfg["save_model"] = args.save_model
    cfg["modes"] = ["bi", "neu"] if args.dataset == "kialo" else ["bi"]
    cfg["tasks"]["main"]["classes"] = ["sup", "att", "neu"] \
                                if args.dataset == "kialo" else ["sup", "att"]

    # Make the log directory if not exists
    os.makedirs(cfg["logs_dir"], exist_ok=True)

    if args.task == "basic":
        if args.logic is not None:
            print("Warning: args.logic will be ignored.")
        cfg["blend_warmup"] = False
        cfg["train_blend_warmup"] = False
        task_combs = [[]]
        blend_type_combs = ["no"]

    elif args.task == "logbert":
        assert args.logic is not None
        cfg["blend_warmup"] = True
        cfg["train_blend_warmup"] = args.train_logic_tasks
        task_combs = [args.logic]
        blend_type_combs = ["no"]

    else:
        raise NotImplementedError()


    for b, (blend_tasks, cfg["blend_type"], trial) in \
                    enumerate(product(task_combs, blend_type_combs, 
                                      list(range(args.n_trials)))):
        tasks = ["main"] + blend_tasks

        if cfg["blend_warmup"]:
            model_path = f'{cfg["models_dir"]}/blend_only-' + \
                                "+".join(sorted(blend_tasks)) + ".model"

            if cfg["train_blend_warmup"]:
                if os.path.exists(model_path): 
                    continue
                os.makedirs(cfg["models_dir"], exist_ok=True)

            else:
                if not os.path.exists(model_path): 
                    print(f"Model not found: {model_path}")
                    continue


        print("\n=============================================================")
        print(f'[{b}]', tasks, cfg["blend_type"])
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


        cfg["prefix"] = prefix = get_prefix()
        cfg["logger"] = logger = get_logger(f'{cfg["logs_dir"]}/{prefix}.log')

        # Prepare the trainer
        trainer = Trainer(cfg)

        print("Loading data...")
        split2batches = {}
        blend_batches = {}
        for task in tasks:
            if task == "main": 
                split2batches = load_data_main(trainer)
            elif task == "nli" and cfg["blend_type"] != "no":
                blend_batches[task] = load_data_nli(trainer)
            elif task == "senti" and cfg["blend_type"] != "no":
                blend_batches[task] = load_data_senti(trainer)
            elif task == "causal" and cfg["blend_type"] != "no":
                blend_batches[task] = load_data_causal(trainer)
            elif task == "normarg" and cfg["blend_type"] != "no":
                blend_batches["normarg_polar"] = \
                        load_data_normarg_polar(trainer)
                blend_batches["normarg_jtype"] = \
                        load_data_normarg_jtype(trainer)
                blend_batches["normarg_norm_senti"] = \
                        load_data_normarg_senti(trainer, ["norm"])
                blend_batches["normarg_conseq_senti"] = \
                        load_data_normarg_senti(trainer, ["conseq","property"])

        cfg["logger"].info("Data")
        for name, batches in list(split2batches.items()) + \
                                list(blend_batches.items()):
            cfg["logger"].info(" - {}: {}".format(
                                name, sum(len(b["id"]) for b in batches)))

        # Warmup with blend tasks
        init_model = None
        if cfg["blend_warmup"]:
            if cfg["train_blend_warmup"]: 
                if os.path.exists(model_path): continue

                trainer.run_epochs_blend_only(blend_batches, 
                                              cfg["n_epochs_blend_only"])
                torch.save(trainer.model.state_dict(), model_path)
                continue
            else:
                if not os.path.exists(model_path): 
                    print("Pretrained model doesn't exist!!")
                    continue
                init_model = torch.load(model_path)

        # Main run
        trainer.run_trials(split2batches["train"], blend_batches,
                            split2batches["val"], split2batches["test"],
                            init_model)
        

        if args.task == "logbert":
            print("")
            for task in cfg["task2classes"].keys():
                if task == "main": continue
                print("Predict:", task)
                trainer.run_epoch(split2batches["test"], "predict", task)
                with open(f'{cfg["logs_dir"]}/{cfg["prefix"]}' + \
                          f'-insts-test-{task}.csv', "w") as f:
                    out_csv = csv.writer(f)
                    out_csv.writerow(["id", "pre_text", "con_text", 
                                      "main_label", "task_label_prob", 
                                      "task_label_pred"])
                    keys = ["id", "pre_text", "con_text", "label", "label_prob",
                            "label_pred"]
                    for batch in split2batches["test"]:
                        for i in range(len(batch["id"])):
                            out_csv.writerow([batch[key][i] for key in keys])
                    


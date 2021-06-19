import random
import os
import json
from csv_utils import *
from time import time
from itertools import product
from collections import defaultdict
import argparse

cfg = {
    "predicates": [],
    "data_dir": None,
    "data_prefix": None,
    "classes": None, 
    "pair_delimiter": None, 
    "modes": None, 
    "default_class": None, 
    "default_wt": 0.2,
    "default_trans_wt": 0.5,
}

pred2info = {
    # feat_file_suffix, n_args, rule
    # Normarg
    "normarg_nneu_8_0": ("normarg", 2, "1: normarg_nneu_8_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu_8_1": ("normarg", 2, "1: normarg_nneu_8_1(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu_8_2": ("normarg", 2, "1: normarg_nneu_8_2(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu_8_3": ("normarg", 2, "1: normarg_nneu_8_3(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu_8_4": ("normarg", 2, "1: normarg_nneu_8_4(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_nneu_8_5": ("normarg", 2, "1: normarg_nneu_8_5(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_nneu_8_6": ("normarg", 2, "1: normarg_nneu_8_6(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_nneu_8_7": ("normarg", 2, "1: normarg_nneu_8_7(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_nneu_4_0": ("normarg", 2, "1: normarg_nneu_4_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu_4_1": ("normarg", 2, "1: normarg_nneu_4_1(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu_4_2": ("normarg", 2, "1: normarg_nneu_4_2(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_nneu_4_3": ("normarg", 2, "1: normarg_nneu_4_3(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_nneu_2_0": ("normarg", 2, "1: normarg_nneu_2_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu_2_1": ("normarg", 2, "1: normarg_nneu_2_1(P, C) -> relation(P, C, 'att') ^2"),

    "normarg_noco_8_0": ("normarg", 2, "1: normarg_noco_8_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_noco_8_1": ("normarg", 2, "1: normarg_noco_8_1(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_noco_8_2": ("normarg", 2, "1: normarg_noco_8_2(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_noco_8_3": ("normarg", 2, "1: normarg_noco_8_3(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_noco_8_4": ("normarg", 2, "1: normarg_noco_8_4(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_noco_8_5": ("normarg", 2, "1: normarg_noco_8_5(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_noco_8_6": ("normarg", 2, "1: normarg_noco_8_6(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_noco_8_7": ("normarg", 2, "1: normarg_noco_8_7(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_noco_4_0": ("normarg", 2, "1: normarg_noco_4_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_noco_4_1": ("normarg", 2, "1: normarg_noco_4_1(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_noco_4_2": ("normarg", 2, "1: normarg_noco_4_2(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_noco_4_3": ("normarg", 2, "1: normarg_noco_4_3(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_noco_2_0": ("normarg", 2, "1: normarg_noco_2_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_noco_2_1": ("normarg", 2, "1: normarg_noco_2_1(P, C) -> relation(P, C, 'att') ^2"),

    "normarg_nneu2_8_0": ("normarg", 2, "1: normarg_nneu2_8_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu2_8_1": ("normarg", 2, "1: normarg_nneu2_8_1(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu2_8_2": ("normarg", 2, "1: normarg_nneu2_8_2(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu2_8_3": ("normarg", 2, "1: normarg_nneu2_8_3(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu2_8_4": ("normarg", 2, "1: normarg_nneu2_8_4(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_nneu2_8_5": ("normarg", 2, "1: normarg_nneu2_8_5(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_nneu2_8_6": ("normarg", 2, "1: normarg_nneu2_8_6(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_nneu2_8_7": ("normarg", 2, "1: normarg_nneu2_8_7(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_nneu2_4_0": ("normarg", 2, "1: normarg_nneu2_4_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu2_4_1": ("normarg", 2, "1: normarg_nneu2_4_1(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu2_4_2": ("normarg", 2, "1: normarg_nneu2_4_2(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_nneu2_4_3": ("normarg", 2, "1: normarg_nneu2_4_3(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_nneu2_2_0": ("normarg", 2, "1: normarg_nneu2_2_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_nneu2_2_1": ("normarg", 2, "1: normarg_nneu2_2_1(P, C) -> relation(P, C, 'att') ^2"),

    "normarg_pall_8_0": ("normarg", 2, "1: normarg_pall_8_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_pall_8_1": ("normarg", 2, "1: normarg_pall_8_1(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_pall_8_2": ("normarg", 2, "1: normarg_pall_8_2(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_pall_8_3": ("normarg", 2, "1: normarg_pall_8_3(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_pall_8_4": ("normarg", 2, "1: normarg_pall_8_4(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_pall_8_5": ("normarg", 2, "1: normarg_pall_8_5(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_pall_8_6": ("normarg", 2, "1: normarg_pall_8_6(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_pall_8_7": ("normarg", 2, "1: normarg_pall_8_7(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_pall_4_0": ("normarg", 2, "1: normarg_pall_4_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_pall_4_1": ("normarg", 2, "1: normarg_pall_4_1(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_pall_4_2": ("normarg", 2, "1: normarg_pall_4_2(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_pall_4_3": ("normarg", 2, "1: normarg_pall_4_3(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_pall_2_0": ("normarg", 2, "1: normarg_pall_2_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_pall_2_1": ("normarg", 2, "1: normarg_pall_2_1(P, C) -> relation(P, C, 'att') ^2"),

    "normarg_situ_8_0": ("normarg", 2, "1: normarg_situ_8_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_situ_8_1": ("normarg", 2, "1: normarg_situ_8_1(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_situ_8_2": ("normarg", 2, "1: normarg_situ_8_2(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_situ_8_3": ("normarg", 2, "1: normarg_situ_8_3(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_situ_8_4": ("normarg", 2, "1: normarg_situ_8_4(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_situ_8_5": ("normarg", 2, "1: normarg_situ_8_5(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_situ_8_6": ("normarg", 2, "1: normarg_situ_8_6(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_situ_8_7": ("normarg", 2, "1: normarg_situ_8_7(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_situ_4_0": ("normarg", 2, "1: normarg_situ_4_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_situ_4_1": ("normarg", 2, "1: normarg_situ_4_1(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_situ_4_2": ("normarg", 2, "1: normarg_situ_4_2(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_situ_4_3": ("normarg", 2, "1: normarg_situ_4_3(P, C) -> relation(P, C, 'att') ^2"),
    "normarg_situ_2_0": ("normarg", 2, "1: normarg_situ_2_0(P, C) -> relation(P, C, 'sup') ^2"),
    "normarg_situ_2_1": ("normarg", 2, "1: normarg_situ_2_1(P, C) -> relation(P, C, 'att') ^2"),


    # NLI
    "nli_20K_ent": ("nli", 2, "1: nli_20k_ent(P, C) -> relation(P, C, 'sup') ^2"),
    "nli_20K_con": ("nli", 2, "1: nli_20k_con(P, C) -> relation(P, C, 'att') ^2"),
    "nli_50K_ent": ("nli", 2, "1: nli_50k_ent(P, C) -> relation(P, C, 'sup') ^2"),
    "nli_50K_con": ("nli", 2, "1: nli_50k_con(P, C) -> relation(P, C, 'att') ^2"),

    # IE conf
    "ie_conf_arg0": ("ie_conf", 2, "1: ie_conf_arg0(P, C) -> relation(P, C, 'att') ^2"),
    "ie_conf_arg1": ("ie_conf", 2, "1: ie_conf_arg1(P, C) -> relation(P, C, 'att') ^2"),
    "ie_conf_arg2": ("ie_conf", 2, "1: ie_conf_arg2(P, C) -> relation(P, C, 'att') ^2"),
    "ie_conf_arg2p": ("ie_conf", 2, "1: ie_conf_arg2p(P, C) -> relation(P, C, 'att') ^2"),
    "ie_conf_any": ("ie_conf", 2, "1: ie_conf_any(P, C) -> relation(P, C, 'att') ^2"),

    # Senti conf
    "sa_conf_np": ("sa_conf", 2, "1: sa_conf_np(P, C) -> relation(P, C, 'att') ^2"),
    "sa_conf_vp": ("sa_conf", 2, "1: sa_conf_vp(P, C) -> relation(P, C, 'att') ^2"),
    "sa_conf_npvp": ("sa_conf", 2, "1: sa_conf_npvp(P, C) -> relation(P, C, 'att') ^2"),
    "sa_cons_np": ("sa_conf", 2, "1: sa_cons_np(P, C) -> relation(P, C, 'sup') ^2"),
    "sa_cons_vp": ("sa_conf", 2, "1: sa_cons_vp(P, C) -> relation(P, C, 'sup') ^2"),
    "sa_cons_npvp": ("sa_conf", 2, "1: sa_cons_npvp(P, C) -> relation(P, C, 'sup') ^2"),

    # Causal
    "causal_3_cause": ("causal", 2, "1: causal_3_cause(P, C) -> relation(P, C, 'sup') ^2"),
    "causal_3_obstruct": ("causal", 2, "1: causal_3_obstruct(P, C) -> relation(P, C, 'att') ^2"),

    "causal_4_cause": ("causal", 2, "1: causal_4_cause(P, C) -> relation(P, C, 'sup') ^2"),
    "causal_4_precede": ("causal", 2, "1: causal_4_precede(P, C) -> relation(P, C, 'sup') ^2"),
    "causal_4_obstruct": ("causal", 2, "1: causal_4_obstruct(P, C) -> relation(P, C, 'att') ^2"),

    "causal_5_cause": ("causal", 2, "1: causal_5_cause(P, C) -> relation(P, C, 'sup') ^2"),
    "causal_5_precede": ("causal", 2, "1: causal_5_precede(P, C) -> relation(P, C, 'sup') ^2"),
    "causal_5_sync": ("causal", 2, "1: causal_5_sync(P, C) -> relation(P, C, 'sup') ^2"),
    "causal_5_obstruct": ("causal", 2, "1: causal_5_obstruct(P, C) -> relation(P, C, 'att') ^2"),

    # Abductive
    "abduct_3_cause": ("abduct", 2, "1: abduct_3_cause(P, C) -> relation(P, C, 'sup') ^2"),
    "abduct_3_obstruct": ("abduct", 2, "1: abduct_3_obstruct(P, C) -> relation(P, C, 'att') ^2"),

    "abduct_4_cause": ("abduct", 2, "1: abduct_4_cause(P, C) -> relation(P, C, 'sup') ^2"),
    "abduct_4_precede": ("abduct", 2, "1: abduct_4_precede(P, C) -> relation(P, C, 'sup') ^2"),
    "abduct_4_obstruct": ("abduct", 2, "1: abduct_4_obstruct(P, C) -> relation(P, C, 'att') ^2"),

    "abduct_5_cause": ("abduct", 2, "1: abduct_5_cause(P, C) -> relation(P, C, 'sup') ^2"),
    "abduct_5_precede": ("abduct", 2, "1: abduct_5_precede(P, C) -> relation(P, C, 'sup') ^2"),
    "abduct_5_sync": ("abduct", 2, "1: abduct_5_sync(P, C) -> relation(P, C, 'sup') ^2"),
    "abduct_5_obstruct": ("abduct", 2, "1: abduct_5_obstruct(P, C) -> relation(P, C, 'att') ^2"),
}



def build_data(out_dir):
    cfg["basic_rules"] = [
        f"{cfg['default_wt']}: relation(P, C, '{cfg['default_class']}') = 1 ^2",
        "relation(P, C, +R) = 1 .",  # relations sum to 1
    ]
    if "trans" in cfg["modes"]:
        cfg["basic_rules"].extend([
            f"{cfg['default_trans_wt']}: transitivity(P,C) & relation(P,Q,'sup') & relation(Q,C,'sup') -> relation(P,C,'sup') ^2",
            f"{cfg['default_trans_wt']}: transitivity(P,C) & relation(P,Q,'att') & relation(Q,C,'att') -> relation(P,C,'sup') ^2",
            f"{cfg['default_trans_wt']}: transitivity(P,C) & relation(P,Q,'sup') & relation(Q,C,'att') -> relation(P,C,'att') ^2",
            f"{cfg['default_trans_wt']}: transitivity(P,C) & relation(P,Q,'att') & relation(Q,C,'sup') -> relation(P,C,'att') ^2",
        ])


    with open(f'{out_dir}/cfg.json', "w") as f:
        f.write(json.dumps(cfg, default=lambda o: str(type(o))))

    data_dir = cfg["data_dir"]
    data_prefix = cfg["data_prefix"]

    # Load train/val/test IDs
    ids = defaultdict(set)
    for mode in cfg["modes"]:
        n_train = 0
        for row in iter_csv_header(f"{data_dir}/{data_prefix}-{mode}-split.csv"):
            if row["split"] == "train":
                split = "ssv" 
                n_train += 1
            else:
                split = row["split"]
            ids[split].add(row["pairid"])
    ids["train"] = ids["sv"] | ids["ssv"]

    # Rules (.psl)
    with open(f'{out_dir}/{data_prefix}.psl', "w") as f:
        for pred in cfg["predicates"]:
            suffix, n_args, rule = pred2info[pred]
            f.write(rule + "\n")
        for rule in cfg["basic_rules"]:
            f.write(rule + "\n")

    # Data specifications (.data)
    for split in ["train", "val", "test"]:
        out_data_dir = f'{out_dir}/data_{split}'
        os.makedirs(out_data_dir, exist_ok=True)
        with open(f'{out_data_dir}/{data_prefix}.data', "w") as f:
            f.write("predicates:\n")
            for pred in cfg["predicates"]:
                suffix, n_args, rule = pred2info[pred]
                f.write(f'  {pred}/{n_args} : closed\n')
            ptype = "open"
            f.write(f'  relation/3 : {ptype}\n')
            if "trans" in cfg["modes"]:
                f.write('  transitivity/2 : closed\n')


            f.write("\nobservations:\n")
            for pred in cfg["predicates"]:
                f.write(f'  {pred} : {pred}_obs.txt\n')
            f.write(f'  relation : relation_obs.txt\n')
            if "trans" in cfg["modes"]:
                f.write(f'  transitivity : transitivity_obs.txt\n')

            f.write("\ntargets:\n")
            f.write(f'  relation : relation_targets.txt\n')

            f.write("\ntruth:\n")
            f.write(f'  relation : relation_truth.txt\n')


    # Data
    pair2label = {}
    for mode in cfg["modes"]:
        for row in iter_csv_header(f'{data_dir}/{data_prefix}-{mode}.csv'):
            if row["relation"] == "1":
                label = "sup"
            elif row["relation"] == "-1":
                label = "att"
            else:
                label = "neu"
            pair2label[row["pairid"]] = label

    for split in ["train", "val", "test"]:
        out_data_dir = f'{out_dir}/data_{split}'
        os.makedirs(out_data_dir, exist_ok=True)

        # Common predicates
        for pred in cfg["predicates"]:
            suffix, n_args, rule = pred2info[pred]
            pairid_set = set()
            with open(f'{out_data_dir}/{pred}_obs.txt', "w") as f:
                for mode in cfg["modes"]:
                    for row in iter_csv_header(f'{data_dir}/{data_prefix}-{mode}-{suffix}.csv'):
                        if row["pairid"] not in ids[split]: continue
                        if row["pairid"] in pairid_set: continue
                        pairid_set.add(row["pairid"])

                        arg1, arg2 = row["pairid"].split(cfg["pair_delimiter"])
                        value = row[pred]
                        f.write(f'{arg1}\t{arg2}\t{value}\n')

        # Transitivity
        if "trans" in cfg["modes"]:
            with open(f'{out_data_dir}/transitivity_obs.txt', "w") as f:
                for row in iter_csv_header(f'{data_dir}/{data_prefix}-trans-split.csv'):
                    if row["split"] != split: continue
                    arg1, arg2 = row["pairid"].split("_")
                    f.write(f'{arg1}\t{arg2}\n')

        # Relation
        if split == "train":
            pairids_inc = {
                "truth": ids[split],
                "obs": ids["sv"],
                "targets": ids["ssv"],
            }
        else:
            pairids_inc = {
                "truth": ids[split],
                "obs": [],
                "targets": ids[split],
            }

        for suffix in ["truth", "obs", "targets"]:
            with open(f'{out_data_dir}/relation_{suffix}.txt', "w") as f:
                for pairid in pairids_inc[suffix]:
                    arg1, arg2 = pairid.split(cfg["pair_delimiter"])
                    label = pair2label[pairid]
                    for cls in cfg["classes"]:
                        if suffix in ["truth", "obs"]:
                            f.write(f'{arg1}\t{arg2}\t{cls}\t{int(cls==label)}\n')
                        else:
                            f.write(f'{arg1}\t{arg2}\t{cls}\n')


def abbr(name):
    tokens = name.split("_")
    if name.startswith("nli") or name.startswith("causal") or name.startswith("abduct"):
        res = tokens[0][:2] + "".join([t[0] for t in tokens[1:-1]]) + \
                tokens[-1][0]
    elif name.startswith("sa_conf"):
        res = "sc" + tokens[-1]
    elif name.startswith("sa_cons"):
        res = "ss" + tokens[-1]
    elif name.startswith("ie_conf"):
        res = "ic" + tokens[-1]
    elif name.startswith("normarg"):
        res = "na" + tokens[1] + tokens[2]
    else:
        raise ValueError(f"Invalid name: {name}")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", choices=["kialo", "debate"], 
                        required=True,
                        help="Dataset.")
    parser.add_argument("-dtype", choices=["normative", "causal"], 
                        required=True,
                        help="Argument type (causal=non-normative).")
    parser.add_argument("-trans", action="store_true",
                        help="Include the chain rules (only for kialo).")
    args = parser.parse_args()

    cfg["data_dir"] = f'../data/{args.dataset}'
    cfg["data_prefix"] = f'{args.dataset}_{args.dtype}_pairs'
    if args.dataset == "kialo":
        cfg["classes"] = ["sup", "att", "neu"]
        cfg["pair_delimiter"] = "_"
        if args.trans:
            cfg["modes"] = ["bi", "neu", "trans"]
            default_trans_wt_comb = [1, 0.5, 0.1]
        else:
            cfg["modes"] = ["bi", "neu"]
            default_trans_wt_comb = [None]

        default_wt_comb = [0.2, 0.3]
        default_class_comb = ["neu"]

        nli_preds_comb = [
            ["nli_50K_ent", "nli_50K_con"], 
        ]
        ie_conf_preds_comb = [
            ["ie_conf_any"], 
        ]
        sa_conf_preds_comb = [
            ["sa_conf_npvp", "sa_cons_npvp"],
        ]
        causal_preds_comb = [
            ["causal_3_cause", "causal_3_obstruct",
             "abduct_3_cause", "abduct_3_obstruct"], 
        ]
        normarg_preds_comb = [
            ["normarg_nneu2_4_0", "normarg_nneu2_4_1", 
             "normarg_nneu2_4_2", "normarg_nneu2_4_3"],
        ]


    else:
        cfg["classes"] = ["sup", "att"]
        cfg["pair_delimiter"] = "#"
        cfg["modes"] = ["bi"]
        cfg["default_class"] = "neu"
     
        default_wt_comb = [0.2, 0.3]
        default_class_comb = ["sup", "att"]
        default_trans_wt_comb = [None]  # Debate

        nli_preds_comb = [
            ["nli_50K_ent", "nli_50K_con"], 
        ]
        ie_conf_preds_comb = [
            ["ie_conf_any"], 
        ]
        sa_conf_preds_comb = [
            ["sa_conf_npvp", "sa_cons_npvp"],
        ]
        causal_preds_comb = [
            ["causal_3_cause", "causal_3_obstruct",
             "abduct_3_cause", "abduct_3_obstruct"], 
        ]
        normarg_preds_comb = [
            ["normarg_nneu2_4_0", "normarg_nneu2_4_1", 
             "normarg_nneu2_4_2", "normarg_nneu2_4_3"],
        ]


    very_start_time = time()

    out_dir_set = set()
    for f, (cfg["default_wt"], cfg["default_trans_wt"], cfg["default_class"],
            nli_preds, ie_conf_preds, sa_conf_preds, 
            causal_preds, normarg_preds) \
            in enumerate(product(default_wt_comb, default_trans_wt_comb,
                                 default_class_comb,
                                 nli_preds_comb, 
                                 ie_conf_preds_comb, sa_conf_preds_comb,
                                 causal_preds_comb, normarg_preds_comb)):

        cfg["predicates"] = nli_preds + ie_conf_preds + sa_conf_preds + \
                            causal_preds + normarg_preds
        if len(cfg["predicates"]) == 0: continue

        start_time = time()
        out_dir = f'{cfg["data_prefix"]}-NC{len(cfg["classes"])}/' + \
                  f'{cfg["data_prefix"]}-NC{len(cfg["classes"])}' + \
                  f'-DC{cfg["default_class"]}' + \
                  f'-PD' + "+".join(sorted(set(
                      [abbr(p) for p in cfg["predicates"]]))) + \
                  f'-WT{cfg["default_wt"]}' + \
                  (f'-TR{cfg["default_trans_wt"]}' if "trans" in cfg["modes"] else '')

        #if os.path.exists(out_dir): continue
        assert out_dir not in out_dir_set
        out_dir_set.add(out_dir)

        print(f"\n[{f}] {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

        print("Building data...")
        build_data(out_dir)

        for split in ["train", "val", "test"]:
            os.makedirs(f'{out_dir}/data_{split}/output', exist_ok=True)
            cmd = f'bash ./run_psl.sh {out_dir} {cfg["data_prefix"]} {split}' + \
                  f' > {out_dir}/data_{split}/output/log.txt'
            print(cmd)
            os.system(cmd)

        print("Time: {:.1f}m".format((time() - start_time)/60))

    print("Total Time: {:.1f}m".format((time() - very_start_time)/60))


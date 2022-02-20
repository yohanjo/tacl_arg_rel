from transformers import BertTokenizerFast, BertModel, AdamW
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from itertools import chain
from collections import defaultdict, Counter
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, FloatTensor
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from time import strftime, localtime, time
import sys
import csv
import json


class BertClassifier(nn.Module):
    def __init__(self, pt_model, task2classes):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(pt_model)
        self.clfs = nn.ModuleDict({
            task: nn.Linear(self.bert.config.hidden_size, len(classes)) \
                for task, classes in task2classes.items()
        })

    def insert_new_clf(self, task, classes):
        if task in self.clfs:
            print(f'Warning: BertClassifier is replacing old clf for "{task}"')
        self.clfs[task] = nn.Linear(self.bert.config.hidden_size, 
                                    len(classes))


    def forward(self, input_ids, attention_mask, token_type_ids, task):
        temb = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )[0][:,0]  # (batch_size, emb_dim)
        logits = self.clfs[task](temb)
        return logits

class Trainer(object):

    def __init__(self, cfg):
        self.cfg = cfg

        # Tokenizer
        self.tok = BertTokenizerFast.from_pretrained(cfg["pt_model"])


    def init_model(self, state_dict=None):
        cfg = self.cfg

        # Tasks and classes
        if "task2classes" not in cfg:
            cfg["task2classes"]["default"] = cfg["classes"]

        self.model = BertClassifier(cfg["pt_model"], cfg["task2classes"])

        if state_dict:
            print("Loading pretrained model...")
            #del state_dict["clfs.main.weight"]
            #del state_dict["clfs.main.bias"]
            #self.model.load_state_dict(state_dict, strict=False)
            self.model.load_state_dict(state_dict)

    def insert_new_clf(self, task, classes):
        self.model.insert_new_clf(task, classes)
        self.cfg["task2classes"][task] = classes

    def run_trials(self, train_batches, blend_batches, val_batches, 
                    test_batches, init_model):
        start_time = time()

        cfg = self.cfg
        logs_dir = cfg["logs_dir"]
        prefix = cfg["prefix"]
        logger = cfg["logger"]
        logger.info(json.dumps(cfg, default=lambda o: str(type(o))))
        pivot_met = cfg["pivot_metric"]

        split2batches = {
            "train": train_batches if train_batches is not None else [],
            "blend": blend_batches,
            "val": val_batches if val_batches is not None else [],
            "test": test_batches if test_batches is not None else [],
        }
        
        best_val_accs, best_test_accs = None, None
        all_val_accs, all_test_accs = defaultdict(list), defaultdict(list)
        for trial in range(cfg["n_trials"]):
            logger.info(f'==================================================')
            logger.info(f'TRIAL {trial+1}')
            logger.info(f'==================================================')
            self.init_model(init_model)
            self.model = self.model.to(cfg["device"])

            if split2batches["train"]:
                self.optim = AdamW(self.model.parameters(),
                                   lr=cfg["learn_rate"],
                                   eps=cfg["epsilon"])

            all_pivot_accs = []
            for epoch in range(cfg["n_epochs"]):
                epoch_start_time = time()
                blend_rate = cfg["blend_rate"][epoch]
                logger.info(f'[Epoch {epoch+1}] blend_rate={blend_rate}')

                if split2batches["train"]:
                    self.run_epoch_blend(split2batches["train"], 
                                         split2batches["blend"], 
                                         blend_rate)
                if split2batches["val"]:
                    val_accs = self.run_epoch(split2batches["val"], "val", "main")
                if split2batches["test"]:
                    test_accs = self.run_epoch(split2batches["test"], "test", "main")
                    if not split2batches["val"]:
                        val_accs = test_accs.copy()

                all_pivot_accs.append(val_accs[pivot_met])
                if best_val_accs is None or \
                        val_accs[pivot_met] > best_val_accs[pivot_met]:
                    best_val_accs = val_accs.copy()
                    best_test_accs = test_accs.copy()

                    # Store best results
                    for batch in split2batches["val"] + split2batches["test"]:
                        batch["label_prob_best"] = batch["label_prob"].copy()
                        batch["label_pred_best"] = batch["label_pred"].copy()

                    # Save inst results to disk
                    for split in ["val", "test"]:
                        if not split2batches[split]: continue
                        out_path = (f'{logs_dir}/{prefix}-insts'
                                    f'-{split}.csv')
                        self.print_inst_results(split2batches[split], out_path)

                    # Save accs to disk
                    for split, accs in [("val", best_val_accs),
                                        ("test", best_test_accs)]:
                        if not split2batches[split]: continue
                        out_path = (f'{logs_dir}/{prefix}-accs'
                                    f'-{split}.csv')
                        self.print_accs(accs, out_path)

                    # Store and save model
                    if split2batches["train"]:
                        logger.info("(Storing model...)")
                        self.best_state_dict = deepcopy(self.model.state_dict())

                        if cfg.get("save_model", False):
                            logger.info("(Saving model to disk...)")
                            model_path = f'{logs_dir}/{prefix}.model'
                            torch.save(self.best_state_dict, model_path)

                logger.info("epoch_time={:.1f}m\n".format(
                                    (time() - epoch_start_time) / 60))

                # Early stop
                if cfg["early_stop"] and \
                        len(all_pivot_accs) > cfg["early_stop"] and \
                        all_pivot_accs[-(cfg["early_stop"]+1)] > \
                                max(all_pivot_accs[-cfg["early_stop"]:]):
                    logger.info("Early stop\n")
                    break

            # Store the best accs for this trial
            for key, score in best_val_accs.items(): 
                all_val_accs[key].append(score)
            for key, score in best_test_accs.items(): 
                all_test_accs[key].append(score)

            
        logger.info("==================================================")
        for split, all_accs in [("val", all_val_accs), ("test", all_test_accs)]:
            logger.info(", ".join(["final_{}_{}={:.3f} ({:.3f})".format(
                                split, met, np.mean(scores), np.std(scores)) \
                            for met, scores in sorted(all_accs.items())]))
            out_path = f'{logs_dir}/{prefix}-accs-{split}-final.csv'
            self.print_final_accs(all_accs, out_path)
        logger.info("total_time: {:.1f}m".format((time() - start_time) / 60))


    def run_epoch(self, batches, mode, task="default"):
        cfg = self.cfg

        if not batches: return {}

        if mode == "train":
            self.model.train()
            torch.set_grad_enabled(True)
            random.shuffle(batches)
            class_wt = self.class_wt(batches, task).to(cfg["device"])
            cfg["logger"].info(f'class_wt={class_wt}')
        else:
            self.model.eval()
            torch.set_grad_enabled(False)
            class_wt = None
        device = cfg["device"]
        label_key = "label"

        losses = []
        for batch in tqdm(batches):
            try:
                logits = self.model(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["input_mask"].to(device),
                            token_type_ids=batch["input_types"].to(device),
                            task=task,
                )  # (batch_size, n_classes)
            except RuntimeError:
                print("[DEBUG] batch_size:", batch["input_ids"].size())
                raise
            
            # Loss
            if mode != "predict":
                labels = LongTensor(batch[label_key]).to(device)
                loss = F.cross_entropy(logits, labels, weight=class_wt)
                losses.append(loss.item())

            # Back prop
            if mode == "train":
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # Labels
            if mode in ["val", "test", "predict"]:
                probs = F.softmax(logits, dim=-1)
                batch[f'{label_key}_pred'] = probs.argmax(dim=-1).tolist()
                batch[f'{label_key}_prob'] = probs.tolist()

            torch.cuda.empty_cache()

        # Accuracy
        accs = {}
        if mode != "predict":
            accs["loss"] = np.mean(losses)
        if mode in ["val", "test"]:
            y_true, y_prob, y_pred = [], [], []
            classes = cfg["task2classes"][task]
            n_classes = len(classes)

            for batch in batches:
                y_true.extend([[int(c==l) for c in range(n_classes)] \
                                    for l in batch[label_key]])
                y_prob.extend(batch[f'{label_key}_prob'])
                y_pred.extend([[int(c==l) for c in range(n_classes)] \
                                    for l in batch[f'{label_key}_pred']])

            # Precision, recall, f1
            accs["prec"], accs["recl"], accs["f1"], _ = \
                    precision_recall_fscore_support(y_true, y_pred)
            for metric in ["prec", "recl", "f1"]:
                for c, cls in enumerate(classes):
                    accs[f'{metric}_{cls}'] = accs[metric][c]
                accs[metric] = np.mean(accs[metric])

            # AUC
            try:
                accs["auc"] = roc_auc_score(y_true, y_prob)
            except ValueError as e:
                print(e)
                print("AUC is set to nan")
                accs["auc"] = float("nan")

            # ACC
            accs["acc"] = accuracy_score(y_true, y_pred)


        if mode != "predict":
            cfg["logger"].info(", ".join(["{}_{}={:.3f}".format(
                                                    mode, met, score) \
                                for met, score in sorted(accs.items())]))
        return accs


    def run_epoch_blend(self, train_batches, blend_batches, blend_rate):
        cfg = self.cfg

        self.model.train()
        torch.set_grad_enabled(True)

        batches = []
        for batch in train_batches:
            batch["task"] = "main"
            batches.append(batch)
        for task, bbatches in blend_batches.items():
            for batch in bbatches:
                batch["task"] = task
                batches.append(batch)
        random.shuffle(batches)
        #class_wt = self.class_wt(train_batches, task).to(cfg["device"])
        #cfg["logger"].info(f'class_wt={class_wt}')

        device = cfg["device"]

        losses = []
        for batch in tqdm(batches):
            task = batch["task"]

            try:
                logits = self.model(
                            input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["input_mask"].to(device),
                            token_type_ids=batch["input_types"].to(device),
                            task=task,
                )  # (batch_size, n_classes)
            except RuntimeError:
                print("[DEBUG] batch_size:", batch["input_ids"].size())
                raise
            
            # Loss
            labels = LongTensor(batch["label"]).to(device)
            loss = F.cross_entropy(logits, labels) #, weight=class_wt)
            losses.append(loss.item())

            # Back prop
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            torch.cuda.empty_cache()

        # Accuracy
        accs = {"loss": np.mean(losses)}
        cfg["logger"].info("\n" + ", ".join(["train_{}={:.3f}".format(
                                                met, score) \
                            for met, score in sorted(accs.items())]))
        return accs


    def run_epochs_blend_only(self, blend_batches, n_epochs):
        cfg = self.cfg
        device = cfg["device"]

        self.init_model()
        self.model.to(device)

        self.model.train()
        torch.set_grad_enabled(True)

        self.optim = AdamW(self.model.parameters(),
                           lr=cfg["learn_rate"],
                           eps=cfg["epsilon"])

        # Flatten all blend batches
        all_batches = []
        for task, batches in blend_batches.items():
            for batch in batches:
                batch["task"] = task
                all_batches.append(batch)

        for epoch in range(n_epochs):
            random.shuffle(all_batches)

            losses = []
            for batch in tqdm(all_batches):
                try:
                    logits = self.model(
                                input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["input_mask"].to(device),
                                token_type_ids=batch["input_types"].to(device),
                                task=batch["task"],
                    )  # (batch_size, n_classes)
                except RuntimeError:
                    print("[DEBUG] batch_size:", batch["input_ids"].size())
                    raise
                
                # Loss
                labels = LongTensor(batch["label"]).to(device)
                loss = F.cross_entropy(logits, labels) #, weight=class_wt)
                losses.append(loss.item())

                # Back prop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                torch.cuda.empty_cache()

        # Accuracy
        accs = {"loss": np.mean(losses)}
        cfg["logger"].info("\n" + ", ".join(["{}={:.3f}".format(
                                                met, score) \
                            for met, score in sorted(accs.items())]))
        return accs


    def make_batches(self, insts, tok_fields):
        cfg = self.cfg

        assert 1 <= len(tok_fields) <= 2
        for inst in insts:
            for field in tok_fields:
                inst[f'{field}_tokens'] = inst[f'{field}_text'].split(" ")
                        #self.tok.tokenize(inst[f'{field}_text'])

        batches = []
        start_idx = 0 
        while start_idx < len(insts):
            input_lens = []
            for batch_size in range(1, min(len(insts)-start_idx+1, 
                                           cfg["max_batch_size"]+1)):
                inst = insts[start_idx + batch_size - 1]
                input_lens.append(min(sum(len(inst[f"{field}_tokens"]) \
                                            for field in tok_fields), 
                                      self.tok.model_max_length))
                if max(input_lens) * len(input_lens) > \
                        cfg["max_n_batch_tokens"]:
                    input_lens = input_lens[:-1]
                    batch_size -= 1
                    break
            subinsts = insts[start_idx:(start_idx + batch_size)]
            batch = {key: [inst[key] for inst in subinsts] \
                        for key in subinsts[0].keys()}

            # Batch input/output ids
            if len(tok_fields) == 1:
                tokens = batch[f"{field}_tokens"]
            else:
                tokens = [(t1, t2) for t1, t2 in \
                                zip(batch[f'{tok_fields[0]}_tokens'], 
                                    batch[f'{tok_fields[1]}_tokens'])]

            tok_enc = self.tok.batch_encode_plus(tokens,
                                                 is_split_into_words=True,
                                                 truncation=True, 
                                                 padding=True)
            batch["input_ids"] = LongTensor(tok_enc[f"input_ids"])
            batch["input_mask"] = LongTensor(tok_enc["attention_mask"])
            batch["input_types"] = LongTensor(tok_enc["token_type_ids"])
            assert batch["input_ids"].size(1) <= 512

            for field in tok_fields:
                del batch[f"{field}_tokens"]

            batches.append(batch)
            start_idx += batch_size
        return batches

    def class_wt(self, batches, task="default"):
        label_key = "label"
        cls2cnt = Counter()
        for batch in batches:
            cls2cnt.update(batch[label_key])
        max_cnt = max(cls2cnt.values())
        class_wt = FloatTensor([max_cnt / cnt for cls, cnt in \
                                    sorted(cls2cnt.items())])
        return class_wt

    def predict(self, text):
        insts = [{ "input_text": text }]
        batches = self.make_batches(insts, ["input"])
        self.run_epoch(batches, "predict")
        return batches[0]["label_prob"][0]
            
    def load_model_from(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.cfg["device"])
        
    def print_inst_results(self, batches, out_path):
        batch = batches[0]
        header = ["id"] + \
                 [key for key in batch.keys() \
                        if key != "id" and \
                            not key.startswith("label") and \
                            not key.startswith("input_")] + \
                 [key for key in batch.keys() if key.startswith("label") and \
                                                not key.endswith("_prob") and \
                                                not key.endswith("_pred")]

        with open(out_path, "w") as f:
            out_csv = csv.writer(f)
            out_csv.writerow(header)
            for batch in batches:
                for i in range(len(batch["id"])):
                    out_csv.writerow([batch[h][i] for h in header])

    def print_accs(self, accs, out_path):
        with open(out_path, "w") as f:
            out_csv = csv.writer(f)
            out_csv.writerow(sorted(accs.keys()))
            out_csv.writerow([score for key, score in sorted(accs.items())])

    def print_final_accs(self, all_accs, out_path):
        with open(out_path, "w") as f:
            out_csv = csv.writer(f)
            out_csv.writerow(chain.from_iterable(
                    [[key, f'{key}_std'] for key in sorted(all_accs.keys())]))
            out_csv.writerow(chain.from_iterable(
                    [[np.mean(scores), np.std(scores)] \
                        for key, scores in sorted(all_accs.items())]))




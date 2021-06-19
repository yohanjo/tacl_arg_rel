# Source Code

## Probabilistic Soft Logic
### Dependencies
* Download [`psl-cli-2.2.2.jar`](https://github.com/linqs/psl) to this directory.

### Run
```
usage: main_psl.py [-h] -dataset {kialo,debate} -dtype {normative,causal} [-trans]

optional arguments:
  -h, --help            show this help message and exit
  -dataset {kialo,debate}
                        Dataset.
  -dtype {normative,causal}
                        Argument type (causal=non-normative).
  -trans                Include the chain rules (only for kialo).
```

To run PSL with the chain rules:
```
$ python main_psl.py -dataset [DATASET] -dtype [DTYPE] -trans
```

To run PSL without the chain rules:
```
$ python main_psl.py -dataset [DATASET] -dtype [DTYPE]
```


## LogBERT
### Dependencies
* PyTorch
* transformers library
* tqdm library

### Run
```
usage: main_sup_blend.py [-h] -task {basic,logbert} -dataset {kialo,debate} -dtype
                         {normative,causal,all}
                         [-logic [{nli,senti,causal,normarg} [{nli,senti,causal,normarg} ...]]]
                         -cuda CUDA [-n_trials N_TRIALS] [-save_model] [-train_logic_tasks]

optional arguments:
  -h, --help            show this help message and exit
  -task {basic,logbert}
                        Task (basic=vanilla BERT, logbert=LogBERT)
  -dataset {kialo,debate}
                        Dataset.
  -dtype {normative,causal,all}
                        Argument type (causal=non-normative).
  -logic [{nli,senti,causal,normarg} [{nli,senti,causal,normarg} ...]]
                        Logic tasks to run.
  -cuda CUDA            CUDA device to use.
  -n_trials N_TRIALS    Number of independent runs.
  -save_model           Save the trained model to disk.
  -train_logic_tasks    Train on logic tasks and save the result model. Otherwise, it's
                        assumed that a saved model already exists.
```


To run LogBERT:
```
$ python main_main_sup_blend.py -task logbert -dataset [DATASET] -dtype [DTYPE] -logic nli senti causal normarg -cuda 0
```

To run basic BERT:
```
$ python main_main_sup_blend.py -task basic -dataset [DATASET] -dtype [DTYPE] -cuda 0
```



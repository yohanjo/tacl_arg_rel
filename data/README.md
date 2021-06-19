# Data
This directory contains two datasets: Kialo and Debatepedia.

## Contents
* `[DATASET]_[TYPE]_pairs-[MODE].csv`: Argument instances.
  * `DATASET`
    * `kialo`: Kialo.
    * `debate`: Debatepedia.
  * `TYPE`
    * `normative`: Normative arguments.
    * `causal`: Non-normative arguments.
  * `MODE`
    * `bi`: Support and attack relations.
    * `neu`: Neutral relations (heuristically-generated).
  * Columns
    * `pairid`: Unique ID for an argument.
    * `argid`: Discussion ID.
    * `propid_to`: Claim ID.
    * `propid_from`: Statement ID.
    * `text_to`: Claim text.
    * `text_from`: Statement text.
    * `relation`
      - `1`: Support.
      - `-1`: Attack.
      - `0`: Neutral.
* `[DATASET]_[TYPE]_pairs-[MODE]-split.csv`: Split information.
  * Columns
    * `pairid`: Unique ID for an argument.
    * `split`:
      - `train`: Fitting (PSL) or training (LogBERT).
      - `val`: Validation.
      - `test`: Test.

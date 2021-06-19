# Annotation
This directory contains the annotation manuals and resulting annotations.

## Annotations (`normarg.csv`)
* `id`: Unique ID of an argument.
* `con`: Claim text.
* `pre`: Statement text
* `rel`: Relation
  * `sup`: Support
  * `att`: Attack
  * `neu`: Neutral (heuristically-generated)
* `con_polar`: Claim's advocacy or opposition (1a)
  * `advocate`: Advocate
  * `object`: Oppose
* `con_P`: Claim's norm target (1b)
* `pre_polar`: Statement's consequence, property, or norm target being advocacy or opposition (3a)
  * `advocate`: Advocate
  * `object`: Oppose
* `pre_senti`: Statement's positivity or negativity (3b)
  * `positive`: Positive
  * `negative`: Negative
* `pre_jtype`: Statement's justification type (2a)
  * `conseq`: Consequence
  * `property`: Property
  * `norm`: Norm
* `pre_Q`: Statement's justification text (2b)

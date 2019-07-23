# Jigsaw-Unintended-Bias-in-Toxicity-Classification
Jigsaw Unintended Bias in Toxicity Classification

## requirements:
- pytorch 1.1.0
- apex
- spacy

## models
- bert-large-uncased (finetuned language model)
- bert-large-cased (finetuned language model)
- bert-large-cased (finetuned language model, trained with uncased data)
- bert-large-wwm-uncased (finetuned language model)
- bert-large-wwm-cased
- bert-base-uncased (finetuned language model)
- GPT-2
- 3*RNN (13 models)

## input
`input` folder should contain following subfolders:
- jigsaw-unintended-bias-in-toxicity-classification: the competition dataset
- pickled-crawl300d2m-for-kernel-competitions
- pickled-glove840b300d-for-10sec-loading
- torch-bert-weights
    - bert-base-cased
    - bert-base-uncased
    - bert-large-cased
    - bert-large-uncased
    - bert-large-uncased-wwm

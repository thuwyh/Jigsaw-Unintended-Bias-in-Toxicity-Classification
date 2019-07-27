# Jigsaw-Unintended-Bias-in-Toxicity-Classification
Since we have rearranged the repo and we do not have enough time to rerun all scripts, there may be some path related errors.
Please kindly modify the code or put the files in the right place required by the code.

## requirements:
- python3
- pytorch 1.1.0
- pytorch-pretrained-bert
- apex
- spacy
- gensim

## models
- bert-large-uncased (finetuned language model)
- bert-large-cased (finetuned language model)
- bert-large-cased (trained with uncased data)
- bert-large-wwm-uncased (finetuned language model)
- bert-large-wwm-cased
- bert-base-uncased (finetuned language model)
- GPT-2
- 3*RNN (13 models)

## input
`input` folder should contain following subfolders:
- jigsaw-unintended-bias-in-toxicity-classification: the competition dataset
- pickled-crawl300d2m-for-kernel-competitions: https://www.kaggle.com/authman/pickled-crawl300d2m-for-kernel-competitions
- pickled-glove840b300d-for-10sec-loading: https://www.kaggle.com/authman/pickled-glove840b300d-for-10sec-loading
- torch-bert-weights
    - bert-base-cased
    - bert-base-uncased
    - bert-large-cased
    - bert-large-uncased
    - bert-large-uncased-wwm
    - bert-large-cased-wwm
    - gpt2
    - bert-base-uncased-vocab.txt
    - bert-large-cased-vocab.txt
    - bert-large-uncased-whole-word-masking-vocab.txt

## training procedure
### prepare data
```bash
cd prepare_data
python make_folds.py
python make_folds_gy.py
python prepare_corpus.py
python prepare_lm_data_cased.py --train_corpus ../input/lm_corpus.txt --output_dir ../input/lm_data_cased/ --epochs_to_generate 1 --max_seq_len 256 --bert_model bert-base-uncased
python prepare_lm_data_uncased.py --train_corpus ../input/lm_corpus.txt --do_lower_case --output_dir ../input/lm_data_uncased/ --epochs_to_generate 1 --max_seq_len 256 --bert_model bert-base-uncased
python prepare_lm_data_wwm.py --train_corpus ../input/lm_corpus.txt --do_lower_case --do_lower_case --output_dir ../input/lm_data_wwm/ --epochs_to_generate 1 --max_seq_len 256 --bert_model bert-base-uncased
python prepare_rnn_data.py
python prepare_rnn_features.py
cd ..
```

### pretrain bert on the competition data

```bash
cd pretrain_bert
python finetune_lm_large_uncased --pregenerated_data ../input/lm_data_uncased/ --bert_model bert-large-uncased --do_lower_case --output_dir ../input/torch-bert-weights/mybert-large-uncased --epochs 1 --fp16 --gradient_accumulation_steps 4
python finetune_lm_large_cased.py --pregenerated_data ../input/lm_data_cased/ --bert_model bert-large-cased --output_dir ../input/torch-bert-weights/mybert-large-cased --epochs 1 --fp16 --gradient_accumulation_steps 4
python finetune_lm_wwm.py --pregenerated_data ../input/lm_data_wwm/ --bert_model bert-large-wwm --do_lower_case --output_dir ../input/torch-bert-weights/mybert-wwm-uncased --epochs 1 --fp16 --gradient_accumulation_steps 4
python finetune_lm_base_uncased.py --pregenerated_data ../input/lm_data_uncased/ --bert_model bert-base-uncased --output_dir ../input/torch-bert-weights/mybert-base-uncased --epochs 1 --fp16 --gradient_accumulation_steps 4
cd ..
```

### train bert models
```bash
cd bert
./train_bert_large_cased.sh
./train_finetuned_bert_base_uncased.sh
./train_finetuned_bert_large_cased.sh
./train-bert-wwmcased.sh
./train_finetuned_bert_wwm_uncased.sh
./train_finetuned_bert_large_uncased.sh
```

### train gpt2 models
```bash
cd gpt2
./train_gpt2.sh
cd ..
```

### train rnn models
We trained 3 different kinds of RNN models during the competition, and they are much similar.
We only include the best one in this repo for simplicity. 
```bash
cd rnn
python train_with_feature_v2.py
cd ..
```
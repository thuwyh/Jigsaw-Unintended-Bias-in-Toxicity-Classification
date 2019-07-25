# Jigsaw-Unintended-Bias-in-Toxicity-Classification
Jigsaw Unintended Bias in Toxicity Classification

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
python finetune_lm_large_uncased --pregenerated_data ../input/lm_data_uncased/ --bert_model bert-large-uncased --do_lower_case --output_dir ../input/torch-bert-weights/mybert_large_uncased --epochs 1 --fp16 --gradient_accumulation_steps 4
python finetune_lm_large_cased.py --pregenerated_data ../input/lm_data_cased/ --bert_model bert-large-cased --output_dir ../input/torch-bert-weights/mybert_large_cased --epochs 1 --fp16 --gradient_accumulation_steps 4
python finetune_lm_wwm.py --pregenerated_data ../input/lm_data_wwm/ --bert_model bert-large-wwm --do_lower_case --output_dir ../input/torch-bert-weights/mybert_wwm --epochs 1 --fp16 --gradient_accumulation_steps 4
python finetune_lm_base_uncased.py --pregenerated_data ../input/lm_data_uncased/ --bert_model bert-base-uncased --output_dir ../input/torch-bert-weights/mybert_base_uncased --epochs 1 --fp16 --gradient_accumulation_steps 4
cd ..
```

### train bert models
```bash
cd bert
./train_bert_large_cased.sh
./train_finetune_bert_base_uncased.sh
./train_finetune_bert_large_cased.sh
./train-bert-wwmcased.sh
python main_bert_yuanhao.py train_all mybert-large-uncased_all --model mybert-large-uncased --lr 0.00005 --batch-size 20 --step 7 --lr_layerdecay 0.98 --split_point 0.25 --kloss 0.02 --clean
python main_bert_yuanhao.py train_all mybert-wwm-uncased_all --model mybert-wwm-uncased --lr 0.00005 --batch-size 20 --step 7 --lr_layerdecay 0.98 --split_point 0.25 --kloss 0.02 --clean
```

### train gpt2 models
```bash
cd gpt2
./train_gpt2.sh
cd ..
```

### train rnn models
We trained 3 different kinds of RNN models, 
```bash
cd rnn
python train_with_feature_v2.py
cd ..
```
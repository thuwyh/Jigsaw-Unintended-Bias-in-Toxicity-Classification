## Datasets
 - https://www.kaggle.com/christofhenkel/torch-bert-weights
 - https://www.kaggle.com/christofhenkel/pytorchpretrainedbert/kernels


## prepare language model data
```bash
python prepare_corpus.py

python prepare_lm_data_uncased.py --train_corpus ../input/lm_corpus.txt --bert_model bert-base-uncased --output_dir ../input/lm_data_uncased/ --do_lower_case --epochs_to_generate 1 --max_seq_len 256
python prepare_lm_data_wwm.py --train_corpus ../input/lm_corpus.txt --bert_model bert-large-uncased-whole-word-masking --do_lower_case --do_whole_word_mask --output_dir ../input/lm_data_wwm/ --epochs_to_generate 1 --max_seq_len 256
python prepare_lm_data_cased.py --train_corpus ../input/lm_corpus.txt --bert_model bert-base-cased --output_dir ../input/lm_data_cased/ --epochs_to_generate 1 --max_seq_len 256
```

## pretrain Bert with competition data
```bash
python finetune_lm_base_uncased --pregenerated_data ../input/lm_data_uncased/ --bert_model bert-base-uncased --do_lower_case --output_dir ../input/mybert_base_uncased --epochs 1 --fp16 --gradient_accumulation_steps 4
python finetune_lm_wwm.py --pregenerated_data ../input/lm_data_wwm/ --bert_model bert-large-wwm --do_lower_case --output_dir ../input/mybert_wwm --epochs 1 --fp16 --gradient_accumulation_steps 4
python finetune_lm_large_cased.py --pregenerated_data ../input/lm_data_cased/ --bert_model bert-large-cased --output_dir ../input/mybert_cased --epochs 1 --fp16 --gradient_accumulation_steps 4
python finetune_lm_base_cased.py --pregenerated_data ../input/lm_data_cased/ --bert_model bert-base-cased --output_dir ../input/mybert_base_cased --epochs 1 --fp16 --gradient_accumulation_steps 4
```

## train
### bert
```bash
python main_bert.py train_all mybert-large-uncased_all --model mybert-large-uncased --lr 0.00005 --batch-size 20 --step 7 --lr_layerdecay 0.98 --split_point 0.25 --kloss 0.02 --clean
python main_bert.py train_all mybert-wwm-uncased_all --model mybert-wwm-uncased --lr 0.00005 --batch-size 20 --step 7 --lr_layerdecay 0.98 --split_point 0.25 --kloss 0.02 --clean
```

### RNN

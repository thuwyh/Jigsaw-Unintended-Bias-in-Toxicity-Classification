## directory structure
- project
    - `input`
        - `jigsaw-unintended-bias-in-toxicity-classification`, competition data
        - `torch-bert-weights`, bert weights, vocab files and configs
            - gpt2
                - config.json
                - merges.txt
                - pytorch_model.bin
                - vocab.json
            - gpt2-vocab.json
    - `src`
    - `experiments`, checkpoints, logs and params

## usage 
### under src_gy

make fold file
- python make_folds.py

gpt2 model( finalall_gpt2_keras )

- . train_gpt2.sh

bert-base-cased model(0623_bert_base_cased_lower)

- . train_bert_base_cased.sh

bert-large-cased model(0623_bert_large_cased_lower)

- . train_bert_large_cased.sh

finetine-bert-base-uncased(0623_finetuned_bert_base_uncased, finetuned bert-base-uncased by yuanhao)

- . train_finetune_bert_base_uncased.sh

finetune-bert-large-cased(finalall_finetuned_bert_large_cased, finetuned bert-large-cased by yuanhao)

bert-whole-word-masking cased model(finalall_wwm_cased)

- .train-bert-wwmcased.sh


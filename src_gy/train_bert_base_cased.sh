python main_bert.py train_all name_for_bertbcmodel --batch-size 48 --lr 0.00005 --step 3 --kloss 0.02 --lr_layerdecay 0.98 --model bert-base-cased --split_point 0.25 --fold_name /folds_weight1.5.pkl --bsample True
mkdir ../kernel_inference_datasets/0623_bert_base_cased_lower/

cp ../experiments/name_for_bertbcmodel/model-0.pt ../kernel_inference_datasets/0623_bert_base_cased_lower/model-0.pt
cp ../input/torch-bert-weights/bert_base_cased/config.json ../kernel_inference_datasets/0623_bert_base_cased_lower/config.json
cp ../input/torch-bert-weights/bert_base_cased/vocab.txt ../kernel_inference_datasets/0623_bert_base_cased_lower/vocab.txt
python main_bert.py train_all name_for_bertlcmodel --batch-size 12 --lr 0.00005 --step 20 --kloss 0.02 --lr_layerdecay 0.98 --model bert-large-cased --split_point 0.25 --fold_name /folds_weight1.5.pkl --bsample True --do_lower_case True
mkdir ../kernel_inference_datasets/0623_bert_large_cased_lower/

cp ../experiments/name_for_bertlcmodel/model-0.pt ../kernel_inference_datasets/0623_bert_large_cased_lower/model-0.pt
cp ../input/torch-bert-weights/bert_large_cased/config.json ../kernel_inference_datasets/0623_bert_large_cased_lower/config.json
cp ../input/torch-bert-weights/bert_large_cased/vocab.txt ../kernel_inference_datasets/0623_bert_large_cased_lower/vocab.txt
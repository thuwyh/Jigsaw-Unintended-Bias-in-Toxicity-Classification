python main_bert.py train_all name_for_fbertlcmodel --batch-size 12 --lr 0.00005 --step 20 --kloss 0.02 --lr_layerdecay 0.98 --model mybertlargecased --split_point 0.25 --fold_name /folds_weight1.5.pkl --bsample True --do_lower_case False
mkdir ../kernel_inference_datasets/finalall_finetuned_bert_large_cased/

cp ../experiments/name_for_fbertlcmodel/model-0.pt ../kernel_inference_datasets/finalall_finetuned_bert_large_cased/model-0.pt
cp ../input/torch-bert-weights/mybertlargecased/config.json ../kernel_inference_datasets/finalall_finetuned_bert_large_cased/config.json
cp ../input/torch-bert-weights/mybertlargecased/vocab.txt ../kernel_inference_datasets/finalall_finetuned_bert_large_cased/vocab.txt
python main_bert.py train_all name_for_fbertbumodel --batch-size 48 --lr 0.00005 --step 3 --kloss 0.02 --lr_layerdecay 0.98 --model mybert-base-uncased --split_point 0.25 --fold_name /folds_weight1.5.pkl --bsample True
mkdir ../kernel_inference_datasets/0623_finetuned_bert_base_uncased/

cp ../experiments/name_for_fbertbumodel/model-0.pt ../kernel_inference_datasets/0623_finetuned_bert_base_uncased/model-0.pt
cp ../input/torch-bert-weights/mybert_base_uncased/config.json ../kernel_inference_datasets/0623_finetuned_bert_base_uncased/config.json
cp ../input/torch-bert-weights/mybert_base_uncased/vocab.txt ../kernel_inference_datasets/0623_finetuned_bert_base_uncased/vocab.txt
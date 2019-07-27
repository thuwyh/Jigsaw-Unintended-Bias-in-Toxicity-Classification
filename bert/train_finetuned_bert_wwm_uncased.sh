python main_bert_yuanhao.py train_all mybert-wwm-uncased_all --model mybert-wwm-uncased --lr 0.00005 --batch-size 20 --step 7 --lr_layerdecay 0.98 --split_point 0.25 --kloss 0.02 --clean
mkdir ../kernel_inference_datasets/mybert-wwm-uncased/

cp ../experiments/mybert-wwm-uncased_all/model-0.pt ../kernel_inference_datasets/mybert-wwm-uncased/model-0.pt
cp ../input/torch-bert-weights/mybert-wwm-uncased/config.json ../kernel_inference_datasets/mybert-wwm-uncased/config.json
cp ../input/torch-bert-weights/mybert-wwm-uncased/vocab.txt ../kernel_inference_datasets/mybert-wwm-uncased/vocab.txt
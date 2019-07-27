python main_bert_yuanhao.py train_all mybert-large-uncased_all --model mybert-large-uncased --lr 0.00005 --batch-size 20 --step 7 --lr_layerdecay 0.98 --split_point 0.25 --kloss 0.02 --clean
mkdir ../kernel_inference_datasets/mybert-large-uncased/

cp ../experiments/mybert-large-uncased_all/model-0.pt ../kernel_inference_datasets/mybert-large-uncased/model-0.pt
cp ../input/torch-bert-weights/mybert-large-uncased/config.json ../kernel_inference_datasets/mybert-large-uncased/config.json
cp ../input/torch-bert-weights/mybert-large-uncased/vocab.txt ../kernel_inference_datasets/mybert-large-uncased/vocab.txt
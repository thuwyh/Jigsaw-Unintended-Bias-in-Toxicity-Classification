python main_gpt2.py train_all name_for_gpt2model --batch-size 48 --lr 0.00005 --step 3 --kloss 0.02 --lr_layerdecay 0.98 --model gpt2 --split_point 0.25 --fold_name /folds_weight1.5.pkl --bsample True
mkdir ../kernel_inference_datasets/finalall-gpt2-keras

cp ../experiments/name_for_gpt2model/model-0.pt ../kernel_inference_dataset/finalall-gpt2-keras/model-0.pt
cp ../input/torch-bert-weights/gpt2/config.json ../kernel_inference_dataset/finalall-gpt2-keras/config.json
cp ../input/torch-bert-weights/gpt2/merge.txt ../input/torch-bert-weights/gpt2/merge.txt
cp ../input/torch-bert-weights/gpt2//vocab.json ../kernel_inference_dataset/finalall-gpt2-keras/vocab.json
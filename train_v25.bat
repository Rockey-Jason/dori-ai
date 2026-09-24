@echo off
setlocal
python dori_ai/data_tools.py
python dori_ai/multilingual_data.py
python build_v25_dataset.py
python train_final.py --data data/corpus/v25_training.txt --retrain-tokenizer --epochs 500 --seq-len 128 --batch-size 8 --dim 64 --heads 4 --layers 4 --ff-dim 256 --lr 0.0003 --vocab-size 768
pause

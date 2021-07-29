#!/bin/bash
epochs=500
modeltype="LSTM_Simple_WO_Regions"


#input_days=50
#output_days=10

#dataset="SL"
#python trainer.py  --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001 --split_date 2021-4-1 --window_slide 1
#python trainer.py  --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001 --split_date 2021-4-1 --window_slide 1
#dataset="JP"
#python trainer.py  --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py  --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#dataset="RUS"
#python trainer.py  --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py  --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#dataset="NOR"
#python trainer.py  --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py  --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001



dataset="Texas NG IT BD KZ KR Germany"
python trainer.py --load_recent --dataset $dataset --epochs 50 --input_days 50 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
python trainer.py --load_recent --dataset $dataset --epochs 100 --input_days 50 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
python trainer.py --load_recent --dataset $dataset --epochs 50 --input_days 50 --output_days 10 --modeltype $modeltype --preprocessing Unfiltered --undersampling None --lr 0.001
python trainer.py --load_recent --dataset $dataset --epochs 100 --input_days 50 --output_days 10 --modeltype $modeltype --preprocessing Unfiltered --undersampling Reduce --lr 0.001


#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001

#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.001
#
#
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.001
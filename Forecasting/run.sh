#!/bin/bash
epochs=50
input_days=14
output_days=7
modeltype="LSTM_Simple_WO_Regions"


dataset="SL"
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling None --lr 0.002
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling Reduce --lr 0.002
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004

dataset="Texas"
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling None --lr 0.002
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling Reduce --lr 0.002
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004

dataset="NG"
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling None --lr 0.002
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling Reduce --lr 0.002
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004

dataset="IT"
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling None --lr 0.002
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling Reduce --lr 0.002
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#!/bin/bash
epochs=150
#input_days=14
#output_days=7
modeltype="LSTM4EachDay_WO_Regions"

dataset="JP Texas IT BD KZ KR Germany"

#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 60 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 40 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 30 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 70 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 20 --modeltype $modeltype --preprocessing Unfiltered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 20 --modeltype $modeltype --preprocessing Unfiltered --undersampling None --lr 0.004

# many ip/op tests to find undersampling is better
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
#
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 10 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 15 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 20 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 25 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days 50 --output_days 30 --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004

#
#dataset="SL"
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling None --lr 0.002
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling Reduce --lr 0.002
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#
#dataset="Texas"
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling None --lr 0.002
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling Reduce --lr 0.002
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#
#dataset="NG"
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling None --lr 0.002
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling Reduce --lr 0.002
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004
#
#dataset="IT"
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling None --lr 0.002
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling None --lr 0.004
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Unfiltered --undersampling Reduce --lr 0.002
#python trainer.py --daily --dataset $dataset --epochs $epochs --input_days $input_days --output_days $output_days --modeltype $modeltype --preprocessing Filtered --undersampling Reduce --lr 0.004

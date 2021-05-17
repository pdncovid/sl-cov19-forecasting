set epochs=50
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype LSTM_Simple_WO_Regions --preprocessing Unfiltered --undersampling None
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype LSTM_Simple_WO_Regions --preprocessing Filtered --undersampling None
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype LSTM_Simple_WO_Regions --preprocessing Unfiltered --undersampling Loss
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype LSTM_Simple_WO_Regions --preprocessing Filtered --undersampling Loss
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype LSTM_Simple_WO_Regions --preprocessing Unfiltered --undersampling Reduce
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype LSTM_Simple_WO_Regions --preprocessing Filtered --undersampling Reduce

python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype LSTM4EachDay_WO_Regions --preprocessing Unfiltered --undersampling None
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype LSTM4EachDay_WO_Regions --preprocessing Filtered --undersampling None
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype LSTM4EachDay_WO_Regions --preprocessing Unfiltered --undersampling Loss
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype LSTM4EachDay_WO_Regions --preprocessing Filtered --undersampling Loss
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype LSTM4EachDay_WO_Regions --preprocessing Unfiltered --undersampling Reduce
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype LSTM4EachDay_WO_Regions --preprocessing Filtered --undersampling Reduce

python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype Dense_WO_regions --preprocessing Unfiltered --undersampling None
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype Dense_WO_regions --preprocessing Filtered --undersampling None
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype Dense_WO_regions --preprocessing Unfiltered --undersampling Loss
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype Dense_WO_regions --preprocessing Filtered --undersampling Loss
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype Dense_WO_regions --preprocessing Unfiltered --undersampling Reduce
python trainer.py --daily --dataset "Sri Lanka" --epochs %epochs% --input_days 14 --output_days 7 --modeltype Dense_WO_regions --preprocessing Filtered --undersampling Reduce

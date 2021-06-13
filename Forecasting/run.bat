set epochs=50
set input_days=14
set output_days=7
set dataset="SL Texas NG IT"

python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days 5 --output_days 10 --modeltype LSTM_Simple_WO_Regions --preprocessing Filtered --undersampling Reduce --lr 0.002
python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days 15 --output_days 10 --modeltype LSTM_Simple_WO_Regions --preprocessing Filtered --undersampling Reduce --lr 0.002
python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days 25 --output_days 10 --modeltype LSTM_Simple_WO_Regions --preprocessing Filtered --undersampling Reduce --lr 0.002
python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days 50 --output_days 10 --modeltype LSTM_Simple_WO_Regions --preprocessing Filtered --undersampling Reduce --lr 0.002


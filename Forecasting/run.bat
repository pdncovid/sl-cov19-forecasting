set epochs=50
set input_days=14
set output_days=7
set dataset="Texas"

python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype LSTM_Simple_WO_Regions --preprocessing Unfiltered --undersampling None
python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype LSTM_Simple_WO_Regions --preprocessing Filtered --undersampling None
python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype LSTM_Simple_WO_Regions --preprocessing Unfiltered --undersampling Loss
python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype LSTM_Simple_WO_Regions --preprocessing Filtered --undersampling Loss
python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype LSTM_Simple_WO_Regions --preprocessing Unfiltered --undersampling Reduce
python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype LSTM_Simple_WO_Regions --preprocessing Filtered --undersampling Reduce

REM python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype LSTM4EachDay_WO_Regions --preprocessing Unfiltered --undersampling None
REM python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype LSTM4EachDay_WO_Regions --preprocessing Filtered --undersampling None
REM python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype LSTM4EachDay_WO_Regions --preprocessing Unfiltered --undersampling Loss
REM python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype LSTM4EachDay_WO_Regions --preprocessing Filtered --undersampling Loss
REM python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype LSTM4EachDay_WO_Regions --preprocessing Unfiltered --undersampling Reduce
REM python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype LSTM4EachDay_WO_Regions --preprocessing Filtered --undersampling Reduce

REM python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype Dense_WO_regions --preprocessing Unfiltered --undersampling None
REM python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype Dense_WO_regions --preprocessing Filtered --undersampling None
REM python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype Dense_WO_regions --preprocessing Unfiltered --undersampling Loss
REM python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype Dense_WO_regions --preprocessing Filtered --undersampling Loss
REM python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype Dense_WO_regions --preprocessing Unfiltered --undersampling Reduce
REM python trainer.py --daily --dataset %dataset% --epochs %epochs% --input_days %input_days% --output_days %output_days% --modeltype Dense_WO_regions --preprocessing Filtered --undersampling Reduce

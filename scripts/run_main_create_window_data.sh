RUN_ARGS="--data_path ../data --file_name df_minimal_clean.csv --save_path ../data --window_size 30 --threshold_val 180"
python main_create_window_data.py $RUN_ARGS

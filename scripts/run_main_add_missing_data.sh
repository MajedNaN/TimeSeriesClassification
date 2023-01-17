RUN_ARGS="--data_path ../data --file_name df_minimal.csv --save_path ../data --window_size 120 --min_diff_multip 1.5 --max_diff_multip 3.2"
python main_add_missing_data.py $RUN_ARGS

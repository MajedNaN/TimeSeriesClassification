RUN_ARGS="--remove_list h2s --data_path ../h5Files --file_name part1.h5 --save_path ../data --flag_minimal --flag_metadata --remove_list h2s --flag_remove_devices --file_name_devices ./devices_to_keep.txt --flag_date_range"
python main_read_store_dataset.py $RUN_ARGS

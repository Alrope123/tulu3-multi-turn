python3 select_dataset.py --dataset_list "allenai/WildChat-1M"
#  "stingning/ultrachat" "allenai/WildChat-1M" "teknium/OpenHermes-2.5" "LDJnr/Capybara" "BAAI/Infinity-Instruct" "allenai/tulu-v2-sft-mixture"
python subsample_tulu_data.py --name ultrachat
python subsample_tulu_data.py --name WildChat-1M
python3 split_jsonl.py --name WildChat-1M --n 100

beaker dataset delete alrope/tulu2-multi-turn
beaker dataset create data/ -n tulu2-multi-turn -w ai2/tulu3-factuality

beaker session create --budget ai2/oe-adapt --gpus 2 --workspace ai2/xinxil-default --image beaker://alrope/xinxil_generate_outputs --mount beaker://alrope/tulu2-multi-turn=/dataset --mount beaker://alrope/xinxil_Meta-Llama-3-70B-Instruct=/model --no-update-default-image --secret-env HF_TOKEN=huggingface_token --bare 

python3 filter.py --input_file /dataset/WildChat-1M.jsonl --download_dir cache --model_name meta-llama/Meta-Llama-3-70B-Instruct --output_file output/filtered_wizard.json --sample_n 50000

python reformatting/reformat_simple2.py --use_demos --model_name meta-llama/Meta-Llama-3-70B-Instruct
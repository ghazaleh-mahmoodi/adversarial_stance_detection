read config_file_name
read score_key
read result_path

echo "train time"  
CUDA_VISIBLE_DEVICES=0 \
python3 train_and_eval_model.py --mode "train" \
--config_file ${config_file_name} \
--trn_data data/twitter_testDT_seenval/train.csv \
--dev_data data/twitter_testDT_seenval/validation.csv \
--score_key f_macro \
--name _${topic} \
--topics_vocab twitter-topic-TRN-semi-sup.vocab.pkl \
--saved_model_file_name ${result_path}
declare -a topics=("DT" "HC" "FM" "LA" "A" "CC")
for topic in  "${topics[@]}"
do
    echo "$topic" 
    echo "train time"  
    CUDA_VISIBLE_DEVICES=0 \
    python3 train_and_eval_model.py --mode "train" \
    --config_file data/config_example_toad.txt \
    --trn_data data/twitter_test${topic}_seenval/train.csv \
    --dev_data data/twitter_test${topic}_seenval/validation.csv \
    --score_key f_macro \
    --name _AdvLearning_${topic} \
    --topics_vocab twitter-topic-TRN-semi-sup.vocab.pkl \
    --mode train 
done
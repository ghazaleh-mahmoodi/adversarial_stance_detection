declare -a topics=("DT" "HC" "FM" "LA" "A" "CC")

for topic in  "${topics[@]}"
do
    echo "$topic" 
    echo "test time"
    python3 train_and_eval_model.py --mode "eval" \
    --config_file data/config_example_toad.txt \
    --trn_data data/twitter_test${topic}_seenval/train.csv \
    --dev_data data/twitter_test${topic}_seenval/test.csv \
    --saved_model_file_name data/checkpoints/ckp-BasicAdv-twitter-example-bicond_${topic}-BEST.tar \
    --topics_vocab glove.twitter.27B.100d.vocabF.pkl \
    --mode eval 
done
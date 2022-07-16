declare -a topics=("FM")

for topic in  "${topics[@]}"
do
    echo "test time"
    python3 train_and_eval_model.py --mode "eval" \
    --config_file data/config_example_toad_${topic}.txt \
    --trn_data data/twitter_test${topic}_seenval/train.csv \
    --dev_data data/twitter_test${topic}_seenval/test_balance.csv \
    --saved_model_file_name data/checkpoints/ckp-BasicAdv-twitter-example-bicond_AdvLearning_${topic}-BEST.tar \
    --topics_vocab twitter-topic-TRN-semi-sup.vocab.pkl \
    --mode eval 
done
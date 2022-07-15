declare -a topics=("DT" "HC" "FM" "LA" "CC" "A")
for topic in  "${topics[@]}"
do
    echo "$topic" 
    echo "test time"
    python3 train_and_eval_model.py --mode "eval" \
    --config_file data/config_example_toad_${topic}.txt \
    --trn_data data/twitter_test${topic}_seenval/train.csv \
    --dev_data data/twitter_test${topic}_seenval/test.csv \
    --saved_model_file_name data/checkpoints/ckp-BasicAdv-twitter-example-bicond_AdvLearning_${topic}-BEST.tar \
    --topics_vocab twitter-topic-TRN-semi-sup.vocab.pkl \
    --mode eval 
done
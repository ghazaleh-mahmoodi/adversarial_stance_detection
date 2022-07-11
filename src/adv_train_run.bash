declare -a topics=("HC" "DT" "FM" "LA" "CC" "A")
declare -a running_mode=("train" "eval")

for topic in  "${topics[@]}"
do
    echo "$topic"
    for mode in "${running_mode[@]}"
    do
        
      CUDA_VISIBLE_DEVICES=0 \
      python3 train_and_eval_model.py --mode ${mode} \
      --config_file data/config_example_toad.txt \
      --trn_data data/twitter_test${topic}_seenval/train.csv \
      --dev_data data/twitter_test${topic}_seenval/validation.csv \
      --score_key f_macro \
      --topics_vocab twitter-topic-TRN-semi-sup.vocab.pkl 
    done
done
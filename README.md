# floret word embeddings for Finnish

This repository contains instructions for training floret word embeddings for Finnish. The embeddings are trained on [CC-100](http://data.statmt.org/cc-100/) corpus (based on CommonCrawl).

[Floret](https://github.com/explosion/floret) is a library for generating word embeddings. It can generate embeddings for any word by using subword information. It's similar to fastText but required a much smaller vector table.

## Training a Finnish floret model

First, [build floret](https://github.com/explosion/floret#build-floret-from-source) from source.

```sh
# Setup a Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install wheel
pip install -r requirements.txt

# Download the CC-100 Finnish corpus and tokenize it
# See http://data.statmt.org/cc-100/
mkdir -p corpus
wget --directory-prefix=corpus http://data.statmt.org/cc-100/fi.txt.xz
xzcat corpus/fi.txt.xz | python -m tokenizer.tokenize_fi > corpus/fi-tokenized.txt

# Train the model
mkdir -p trained_models
floret skipgram -mode floret -hashCount 2 -bucket 50000 -minn 3 -maxn 5 -minCount 50 \
    -dim 300 -epoch 5 -thread 16 -input corpus/fi-tokenized.txt \
	-output trained_models/fi-300-50k-minn3-maxn5
```

## Quality

This section shows results of comparisons of the trained floret embeddings on downstream tasks against the [Finnish fastText embeddings](https://fasttext.cc/docs/en/crawl-vectors.html) published by Facebook.

See [instructions for running the evaluations](evaluation.md).

| Task                                          | fastText | floret |
| --------------------------------------------- | -------- | ------ |
| Text classification (eduskunta-vkk), F1 macro | 0.43     | 0.41   |
| Correlation with morphology (QVEC)            | 0.652    | 0.640  |

Higher values are better.

## Trained model file sizes

| Model            | Size (uncompressed) |
| ---------------- | ------------------- |
| fastText (.bin)  | 6.8 GB              |
| floret (.floret) | 0.1 GB              |

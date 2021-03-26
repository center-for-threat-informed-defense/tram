# Machine Learning Model

The machine learning based approach uses [SRoBERTa](https://huggingface.co/sentence-transformers/stsb-roberta-large) as a sentence encoder. These sentence embedddings are used as a feature for a downstream one-vs-rest binary classifier for each of the TTPs outlined in ATT&CK. 

On the first run, this model is downloaded to disk and cached in `~/.cache/torch/sentence_transformers/sbert.net_models_stsb-roberta-large`.

As a caveat: because of the limited amount of available labeled (sentence, techniques) pairs, the performance of this model is expected to be quite poor initially. As more gold-standard reports are labeled by experts, the results are expected to improve.

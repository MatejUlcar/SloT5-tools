# SloT5-tools
Scripts used for training and evaluation of SloT5 models

## Training
Corpora used for training the SloT5 models is the same as for SloBERTa model. For pre-processing the corpora, please refer to https://github.com/clarinsi/Slovene-BERT-Tool
For SloT5 we just reformat the txt files (before sentencepiece tokenization) into TSV format, using `training/txt2tsv.py`

If we have a nvidia enroot container, with `text-to-text-transfer-transformer` installed in the container, we can run the pre-training with the `training/t5_pretraining.sh` script, where we provide
the desired `.gin` files, containing the model architecture and other parameters.

## Evaluation
For evaluation, we use the provided `evaluation/run_summarization.py` code by [Huggingface](https://github.com/huggingface/transformers). For each evaluation task, a bash script is provided in
the `evaluation` folder with the parameters used for fine-tuning the T5 models.

After fine-tuning, we can calculate the F1 and accuracy scores for each classification task using the `evaluation/t5-predictions-analysis.py` script.

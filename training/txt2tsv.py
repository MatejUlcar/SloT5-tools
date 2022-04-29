for split in ['eval', 'train']:
    corpus='corpora/sl_corpora_all.roberta.'+split
    output='corpora/sl_corpora_all.t5.'+split+'.tsv'
    with open(corpus, 'r') as reader, open(output, 'w') as writer:
        for line in reader:
            line = line.replace('\t', ' ')
            writer.write('\t'+line)


from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import csv
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--eval_file", required=True, help="csv/jsonl eval file, prepared for T5 finetuning")
parser.add_argument("--label", help="if jsonl, provide the key/label of prediction")
parser.add_argument("--predictions", required=True, help="predictions as output by T5 finetuning")
parser.add_argument("--labelmap", default=False, help="Optional, what label in eval file should match with what label in predictions. Expects a tsv, with eval file labels in first column and matching prediction labels in second column.")
parser.add_argument("--wer", action="store_true", help="Naive WER and SER scores, not considering alignments, just direct pair-wise comparison, ignoring the following punctuation marks: .!?:,;")
parser.add_argument("--ner", action="store_true", help="Adjust scores for NER")
args = parser.parse_args()


def wer(y_ev, y_pr):
    ser = 0
    wer = 0
    totwords = 0
    totsents = len(y_ev)
    for s in range(totsents):
        goldsent = y_ev[s].strip().replace('.','').replace('!','').replace('?','').replace(':','').replace(',','').replace(';','')
        predsent = y_pr[s].strip().replace('.','').replace('!','').replace('?','').replace(':','').replace(',','').replace(';','')
        if goldsent != predsent:
            ser += 1
        goldwords = goldsent.split()
        predwords = predsent.split()
        totwords += len(goldwords)
        for w in range(min(len(goldwords),len(predwords))):
            if goldwords[w] != predwords[w]:
                wer += 1
        if len(predwords) < len(goldwords):
            wer += len(goldwords) - len(predwords)
    return {'wer': wer/totwords, 'ser': ser/totsents}

def nerls(y_ev, y_pr):
    y_ev_i = []
    y_pr_i = []
    for s in range(len(y_ev)):
        sent_ev = y_ev[s].replace('B-','').replace('I-','').strip().split()
        sent_pr = y_pr[s].replace('B-','').replace('I-','').strip().split()
        for w in range(len(sent_ev)):
            y_ev_i.append(sent_ev[w])
            y_pr_i.append(sent_pr[w] if w < len(sent_pr) else 'O')
    return y_ev_i, y_pr_i

def ner2(x_ev, y_ev, y_pr):
    results = {'osebe': {'tp':0, 'fp':0, 'tn':0, 'fn':0},
               'lokacije': {'tp':0, 'fp':0, 'tn':0, 'fn':0},
               'organizacije': {'tp':0, 'fp':0, 'tn':0, 'fn':0}
              }
    for i in range(len(y_ev)):
        entity_category = x_ev[i].split(':')[0]
        golden_entities = y_ev[i].split(',')
        golden_entities = [g.strip() for g in golden_entities]
        predicted_entities = y_pr[i].split(',')
        predicted_entities = [p.strip() for p in predicted_entities]
        for gld in golden_entities:
            if gld == 'brez':
                continue
            elif gld in predicted_entities:
                results[entity_category]['tp'] += 1
            else:
                results[entity_category]['fn'] += 1
        for prd in predicted_entities:
            if prd in golden_entities:
                continue
            else:
                results[entity_category]['fp'] += 1
    return results

y_ev = []
x_ev = []
if args.eval_file.endswith('.csv'):
    with open(args.eval_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        for line in reader:
            y_ev.append(line[1])
            x_ev.append(line[0])
elif args.eval_file.endswith('.json'):
    with open(args.eval_file) as f:
        for line in f:
            line = json.loads(line.strip())
            y_ev.append(line[args.label])
            #x_ev.append(line['text'])
print(len(y_ev))
if args.labelmap:
    labelmap = {}
    with open(args.labelmap, 'r') as reader:
        for line in reader:
            line = line.strip().split('\t')
            labelmap[line[1]] = line[0]

y_pr = []
with open(args.predictions, 'r') as reader:
    linecount = 0
    for line in reader:
        linecount += 1
        line = line.strip('\n')
        line = line.replace('<extra_id_0>','')
        if args.labelmap:
            y_pr.append(labelmap[line])
        else:
            y_pr.append(line)
print(len(y_pr), linecount)
assert len(y_ev)==len(y_pr)
if args.nerls:
    y_ev, y_pr = nerls(y_ev, y_pr)
    print("F1 by class:", f1_score(y_ev, y_pr, average=None))
if args.ner:
    nerscore = ner2(x_ev, y_ev, y_pr)
    nerf1 = [nerscore[x]['tp']/(nerscore[x]['tp']+0.5*(nerscore[x]['fp']+nerscore[x]['fn'])) for x in nerscore]
    print("precision", [nerscore[x]['tp']/(nerscore[x]['tp']+nerscore[x]['fp']) for x in nerscore])
    print("recall", [nerscore[x]['tp']/(nerscore[x]['tp']+nerscore[x]['fn']) for x in nerscore])
    print("F1 by class:", nerf1, "macro F1:", sum(nerf1)/3)

#print(confusion_matrix(y_ev, y_pr))
print("F1 micro:", f1_score(y_ev, y_pr, average='micro'))
print("F1 macro:", f1_score(y_ev, y_pr, average='macro'))
print("Acc:", accuracy_score(y_ev, y_pr))
if args.wer:
    werscore = wer(y_ev, y_pr)
    print("WER:", werscore['wer'], "WRR:", 1-werscore['wer'])
    print("SER:", werscore['ser'], "SRR:", 1-werscore['ser'])

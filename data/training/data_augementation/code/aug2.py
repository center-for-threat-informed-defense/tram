# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from eda import *
import sys
import json

#arguments to be parsed from command line
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=False, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float, help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted")
args = ap.parse_args()

#the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join("dirname(args.input)", 'eda_' + "basename(args.input)")

#number of augmented sentences to generate per original sentence
num_aug = 9 #default
if args.num_aug:
    num_aug = args.num_aug

#how much to replace each word by synonyms
alpha_sr = 0.1#default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

#how much to insert new words that are synonyms
alpha_ri = 0.1#default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

#how much to swap words
alpha_rs = 0.1#default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

#how much to delete words
alpha_rd = 0.1#default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')
#generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd):
    all_sent={}
    len_list=[]
    file_path="//Users/safe/Desktop/screeshot_dec_and data/data_aug/eda_nlp-master/all_analyzed_reports.json"
    with open(file_path,"r") as f:
        data_content=json.load(f)
    for obj in list(data_content.keys()):
        len_list.append(len(data_content[obj]))
    max_sent=max(len_list)
    for objp in list(data_content.keys()):
        if (len(data_content[objp]))==max_sent:
            print(objp)
    # max_sent=max(len_list)
    for key in list(data_content.keys()):
        if key.endswith("-multi"):
            all_sent[key]=data_content[key]
            pass
        else:
            tech_sent_list=[]
            tech_sent=data_content[key]
            tech_len=len(tech_sent)
            multiplier=(max_sent/tech_len)
            diff=multiplier-int(multiplier)
            if diff==0:
                num=int(multiplier)
            else:
                if diff>=0.5:
                    num=int(multiplier)+1
                else:
                    num=int(multiplier)
            print(max_sent,tech_len,key," .  ",num)
            for sentence in tech_sent:
                if sentence!="":
                    tech_sent_list.append(sentence)
                    aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num)
                    tech_sent_list.extend(aug_sentences)
                all_sent[key]=tech_sent_list
    with open(
                "/Users/safe/Desktop/data_aug/eda_nlp-master/res3.json",
                "w",
            ) as json_file:
                json.dump(all_sent, json_file, indent=4)    
gen_eda(args.input, output, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd)
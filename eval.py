from refer import REFER
import pandas as pd
import os
import json
import os.path as osp
import argparse
import pickle
#  sys.path.insert(0, './evaluation')
from evaluation.refEvaluation import RefEvaluation


def read_file(file, file_type='json'):
    assert file_type in ['json', 'pkl']
    if file_type == 'json':
        with open(file, 'r') as f:
            data = json.load(f)
    elif file_type == 'pkl':
        with open(file, 'rb') as f:
            data = pickle.load(f)
    return data

def reformat(json_content, anns2refs_dict, ann_key_name='ann_id', re_key_name='generated'):

    return [{'ref_id': anns2refs_dict[e[ann_key_name]], 'sent': e[re_key_name]}
            for e in json_content]

def evaluate_file(file, refer, anns2refs_dict, file_type='json', re_key_name='generated'):

    data = read_file(file, file_type)
    reformat_data = reformat(data, anns2refs_dict, re_key_name=re_key_name)
    refEval = RefEvaluation(refer, reformat_data)
    refEval.evaluate()

    return refEval.eval


def evaluate_files(args, refer, anns2refs_dict):

    results = pd.DataFrame()

    file_suffix = '.' + args.file_type
    for file in [f for f in os.listdir(args.input_dir) if f.endswith(file_suffix)]:
        infile = osp.join(args.input_dir, file)
        eval_results = evaluate_file(infile, refer, anns2refs_dict, args.file_type, args.re_key_name)
        results = results.append(pd.Series(eval_results, name=file.replace(file_suffix, '')))

    results.to_csv(osp.join(args.out_dir, args.dataset+'_quality_results.csv'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root',
                    default='/home/simeon/Dokumente/Datasets/ReferIt/')

    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--splitBy', default='unc')
    parser.add_argument('--input_dir')
    parser.add_argument('--file_type', default='json', choices=['json', 'pkl'])
    parser.add_argument('--re_key_name', default='generated')
    parser.add_argument('--out_dir', default=None)

    args = parser.parse_args()

    if not args.out_dir:
        args.out_dir = osp.dirname(os.path.abspath(__file__))

    refer = REFER(args.data_root, args.dataset, args.splitBy)
    anns2refs = {value['ann_id']: value['ref_id'] for value in refer.annToRef.values()}

    evaluate_files(args, refer, anns2refs)

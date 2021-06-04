from refer import REFER
import pandas as pd
import os
import json
import os.path as osp
import argparse
#  sys.path.insert(0, './evaluation')
from evaluation.refEvaluation import RefEvaluation


def reformat(json_content, anns2refs_dict):

    return [{'ref_id': anns2refs_dict[e['target']], 'sent': e['caption']}
            for e in json_content]


def evaluate_file(file, refer, anns2refs_dict):

    with open(file, 'r') as f:
        data = json.load(f)
    reformat_data = reformat(data, anns2refs_dict)
    refEval = RefEvaluation(refer, reformat_data)
    refEval.evaluate()

    return refEval.eval


def evaluate_files(args, refer, anns2refs_dict):

    results = pd.DataFrame()

    for file in os.listdir(args.input_dir):
        infile = osp.join(args.input_dir, file)
        eval_results = evaluate_file(infile, refer, anns2refs_dict)
        results = results.append(pd.Series(eval_results, name=file))

    results.to_csv(osp.join(args.out_dir, args.dataset+'_quality_results.csv'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root',
                    default='/home/simeon/Dokumente/Datasets/')

    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--splitBy', default='unc')
    parser.add_argument('--input_dir')
    parser.add_argument('--out_dir', default=None)

    args = parser.parse_args()

    if not args.out_dir:
        args.out_dir = osp.dirname(os.path.abspath(__file__))

    refer = REFER(args.data_root, args.dataset, args.splitBy)
    anns2refs = {value['ann_id']: value['ref_id'] for value in refer.annToRef.values()}

    evaluate_files(args, refer, anns2refs)

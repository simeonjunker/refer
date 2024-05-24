from refer import REFER
import pandas as pd
import os
import json
import os.path as osp
import argparse
import pickle
#  sys.path.insert(0, './evaluation')
from evaluation.refEvaluation import RefEvaluation
from glob import glob

def parse_filename(file_path):
    
    file_name = osp.split(file_path)[-1]
    file_stem = osp.splitext(file_name)[0]
    split_filename = file_stem.split('_')
    file_idx = None
    if len(split_filename) == 7:        
        dataset, split, architecture, context, noise, epoch, _ = split_filename
    elif len(split_filename) == 8: 
        dataset, split, architecture, context, noise, epoch, _, file_idx = split_filename
    context = context.split(':')[-1]
    noise = noise.split(':')[-1]
    noise = float(noise.replace('-', '.'))
    epoch = epoch.split(':')[-1]
    
    return dataset, split, architecture, context, noise, epoch, file_idx

def read_file(file):
    with open(file, 'r') as f:
        generated = json.load(f)                 
    return generated

def reformat(json_content, anns2refs_dict, ann_key_name='ann_id', re_key_name='generated'):

    return [{'ref_id': anns2refs_dict[e[ann_key_name]], 'sent': e[re_key_name]}
            for e in json_content]

def evaluate_file(file, refer, anns2refs_dict, re_key_name='generated'):

    data = read_file(file)
    reformat_data = reformat(data, anns2refs_dict, re_key_name=re_key_name)
    refEval = RefEvaluation(refer, reformat_data)
    refEval.evaluate()

    return refEval.eval


def evaluate_files(args, refer, anns2refs_dict):

    # init df
    results = pd.DataFrame()
    
    # find files
    
    q = osp.join(
        #args.input_dir, args.dataset, args.architecture, '**' , 
        args.dataset + '_' + args.split.lower() + '*_generated*.json')
    print q
    files = sorted(glob(q))
    print '\n'
    print 'found files:'
    for f in files:
        print osp.split(f)[-1]
    print '\n'
    
    for file in files:
        print 'process file ' + file
        # evaluate
        eval_results = evaluate_file(file, refer, anns2refs_dict, 'generated')

        # add system info        
        dataset, split, architecture, context, noise, _, file_idx = parse_filename(file)
        for key, value in zip(
            ['dataset', 'split', 'architecture', 'context', 'noise', 'file_idx'], 
            [dataset, split, architecture, context, noise, file_idx]
            ):
            eval_results[key] = value
        
        # get system name
        filename = osp.split(file)[-1]
        file_base = osp.splitext(filename)[0]
        system_name = file_base.split('_generated.')[0]
        
        # add to df
        results = results.append(pd.Series(eval_results, name=system_name))

    # write to csv
    outfile_path = osp.join(args.out_dir, args.architecture + '_' + args.dataset + '_' + args.split + '_metrics.csv')
    print "write results to " + outfile_path
    results.to_csv(outfile_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root',
                    default='/home/simeon/Dokumente/Datasets/ReferIt/')

    parser.add_argument('--input_dir')
    parser.add_argument('--out_dir', default=None)
    parser.add_argument('--dataset', default='refcoco')
    parser.add_argument('--architecture', choices=['TRF', 'CLIP_GPT'])
    parser.add_argument('--split', default='val', choices=['testA', 'testB', 'val'])
    parser.add_argument('--splitBy', default='unc')

    args = parser.parse_args()

    if not args.out_dir:
        args.out_dir = osp.dirname(os.path.abspath(__file__))

    refer = REFER(args.data_root, args.dataset, args.splitBy)
    anns2refs = {value['ann_id']: value['ref_id'] for value in refer.annToRef.values()}

    evaluate_files(args, refer, anns2refs)

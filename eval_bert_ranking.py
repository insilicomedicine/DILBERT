from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

from data_utils.utils import read_dataset, read_clinical_trials_dataset
from models.bert_ranker import RankingMapper
from typing import List, Dict, Any
from copy import deepcopy


Entity = Dict[str, Any]


def check_label(predicted_cui: str, golden_cui: str) -> int:
    """
    Comparing composite concept_ids
    """
    return len(set(predicted_cui.replace('+', '|').split("|")).
               intersection(set(golden_cui.replace('+', '|').split("|")))) > 0


def is_correct(meddra_code: str, candidates: List[str], topk: int = 1) -> int:
    for candidate in candidates[:topk]:
        if check_label(candidate, meddra_code): return 1
    return 0


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model_dir')
    parser.add_argument('--data_folder')
    parser.add_argument('--vocab')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--ct_dataset', action='store_true')
    parser.add_argument('--out_of_kb', action='store_true')
    return parser.parse_args()


def eval_splitted_entities(predicted_entities: List[Entity], gold_entities: List[Entity]) -> float:
    gold_entities = pd.DataFrame(gold_entities)
    correct_top1 = []
    for pred_entity in predicted_entities:
        entity_id = pred_entity['entity_id'].split('_')[0]
        labels = gold_entities[gold_entities.entity_id == entity_id]['label'].iloc[0].split(',')
        for label in labels:
            is_pred_correct = is_correct(label, pred_entity['label'], topk=1)
            if is_pred_correct:
                correct_top1.append({'entity_id': entity_id, 'is_corr': is_pred_correct})
                break
    correct_top1 = pd.DataFrame(correct_top1)
    return correct_top1.groupby('entity_id')['is_corr'].agg('min').mean()


def eval_entities(predicted_entities: List[Entity], gold_entities: List[Entity]) -> float:
    correct_top1 = []
    for gold_entity, pred_entity in tqdm(zip(gold_entities, predicted_entities), total=len(gold_entities)):
        predicted_top_labels = pred_entity['label']
        label = gold_entity['label']
        correct_top1.append(is_correct(label, predicted_top_labels, topk=1))
    return np.mean(correct_top1)


if __name__ == '__main__':
    args = get_arguments()
    """if args.ct_dataset:
        entities = read_clinical_trials_dataset(args.data_folder, args.out_of_kb)
    else:"""
    entities = read_dataset(args.data_folder)
    bert_ranker = RankingMapper(args.model_dir, args.vocab)
    predicted = bert_ranker.predict(entities)
    pickle.dump( predicted, open( "predicted.p", "wb" ) )
    """if args.split:
        acc_1 = eval_splitted_entities(predicted, entities)
    else:
        acc_1 = eval_entities(predicted, entities)"""

    print(f"Acc@1 is {acc_1}")


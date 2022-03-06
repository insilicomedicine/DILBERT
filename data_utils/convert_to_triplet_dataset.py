from models.bert_ranker import RankingMapper
from models.random_sampler import RandomSampler
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from hierarchy import MeSHGraph, UMLS_hierarchy
import os
from glob import glob

from typing import Optional, Any, List, Tuple


def parse_line(line: str):
    splitted_line = line.split('||')
    return splitted_line[-2], splitted_line[-1]


def read(folder: str):
    data = []
    pattern = os.path.join(folder, '*.concept')
    for fpath in glob(pattern):
        with open(fpath, encoding='utf-8') as input_stream:
            for line in input_stream:
                mention, concept_id = parse_line(line.strip())
                data.append([mention, concept_id])
    return pd.DataFrame(data, columns=['entity_text', 'label'])


def is_equal(label_1: str, label_2: str) -> bool:
    """
    Comparing composite concept_ids
    """
    return len(set(label_1.replace('+', '|').split("|")).intersection(set(label_2.replace('+', '|').split("|")))) > 0


def get_hierarchy_aware_negatives(hierarchy: Any, label: str) -> List[str]:
    parents: List[Any] = hierarchy.get_parents(label)
    negatives: List[str] = []
    for parent in parents:
        children = hierarchy.get_children(parent)
        children = [child for child in children if not is_equal(child, label)]
        negatives += children
    return negatives


def find_last_occurence(ordered_labels: List[Tuple[str, str]], label: str) -> int:
    for i in range(len(ordered_labels) - 1, 1, -1):
        if is_equal(ordered_labels[i][0], label): return i
    return 0


def find_first_occurence(ordered_labels: List[Tuple[str, str]], label: str) -> int:
    for i, item in enumerate(ordered_labels):
        if is_equal(item[0], label): return i
    return i


def get_negative_examples(label: str, negatives_count: int, hierarchy: Optional[Any] = None,
                          hierarchy_aware: bool = True, ordered_labels: Optional[List[str]] = None) -> List[str]:
    if hierarchy_aware:
        subsample = get_hierarchy_aware_negatives(hierarchy, label)
        ordered_subsample = [(concept_id, concept_name) for concept_id, concept_name in ordered_labels
                             if concept_id in subsample and not is_equal(concept_id, label)]
    else:
        last_idx = find_last_occurence(ordered_labels, label)
        ordered_subsample = ordered_labels[last_idx + 1:]

    ordered_subsample = [(concept_id, concept_name) for concept_id, concept_name in ordered_subsample
                         if not is_equal(concept_id, label)]

    if hierarchy_aware:
        parents = hierarchy.get_parents(label)
        ordered_subsample = [(concept_id, concept_name) for concept_id, concept_name in ordered_subsample if concept_id not in parents]
    negatives = [concept_name for concept_id, concept_name in ordered_subsample[:negatives_count]]
    return negatives


def get_positive_examples(label: str, positives_count: int, hierarchy: Optional[Any] = None,
                          parents_count: int = 0, ordered_labels: Optional[List[str]] = None) -> List[str]:

    positives = [concept_name for concept_id, concept_name in ordered_labels if is_equal(concept_id, label)]
    positives = positives[:positives_count]
    parents = []
    if parents_count > 0:
        parent_labels = hierarchy.get_parents(label)
        parents = [concept_name for concept_id, concept_name in ordered_labels if concept_id in parent_labels]
    return positives + parents


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_data')
    parser.add_argument('--vocab')
    parser.add_argument('--hierarchy')
    parser.add_argument('--negatives_count_per_example', type=int, default=30)
    parser.add_argument('--positive_count_per_example', type=int, default=15)
    parser.add_argument('--parents_sample_count', type=int, default=0)
    parser.add_argument('--save_to')
    parser.add_argument('--path_to_bert_model')
    parser.add_argument('--hard', action='store_true')
    parser.add_argument('--hierarchy_aware', action='store_true')
    parser.add_argument('--UMLS', action='store_true')
    args = parser.parse_args()

    data = read(args.input_data)
    hierarchy = None
    if args.hierarchy_aware and args.UMLS:
        hierarchy = UMLS_hierarchy(args.hierarchy)
    elif args.hierarchy_aware:
        hierarchy = MeSHGraph(args.hierarchy)
    if args.hard:
        model = RankingMapper(model_path=args.path_to_bert_model, vocab_path=args.vocab)
        model.search_count = 1024
    else:
        model = RandomSampler(vocab_path=args.vocab, search_count=1024)
    batch_size = 32
    with open(args.save_to, 'w', encoding='utf-8') as output_stream:
        for batch_start in tqdm(range(0, data.shape[0], batch_size), total=data.shape[0]//batch_size):
            batch_end = min(data.shape[0], batch_start + batch_size)
            batch_texts = data.entity_text[batch_start:batch_end].tolist()
            batch_labels = data[batch_start:batch_end].label.tolist()
            if args.hard:
                batch_nearest_concepts = model.get_candidates(batch_texts)
            else:
                batch_nearest_concepts = model.get_candidates(batch_labels)
            for entity_text, label, concepts in zip(batch_texts, batch_labels, batch_nearest_concepts):
                nearest_concept_ids, _, nearest_concept_names = concepts
                ordered_labels = [(concept_id, concept_name) for concept_id, concept_name in
                                  zip(nearest_concept_ids, nearest_concept_names)]
                positive_examples = get_positive_examples(label, args.positive_count_per_example, hierarchy,
                                                      args.parents_sample_count, ordered_labels)
                negative_examples = get_negative_examples(label, args.negatives_count_per_example, hierarchy,
                                                      args.hierarchy_aware, ordered_labels)

                for positive_example in positive_examples:
                    if entity_text == positive_example: continue
                    for negative_example in negative_examples:
                        output_stream.write(f'{entity_text}\t{positive_example}\t{negative_example}\n')


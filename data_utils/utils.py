import pandas as pd
import os
from glob import glob
from typing import Dict, List, Any


Entity = Dict[str, Any]


def read_vocab(path: str) -> pd.DataFrame:
    data = []
    with open(path, encoding='utf-8') as input_stream:
        for line in input_stream:
            data.append({'label': line.split('||')[0], 'concept_name':line.strip().split('||')[1]})
    return pd.DataFrame(data)


def read_annotation_file(ann_file_path: str) -> List[Entity]:
    data = []
    with open(ann_file_path, encoding='utf-8') as input_stream:
        for row_id, line in enumerate(input_stream):
            splitted_line = line.strip().split('||')
            mention = splitted_line[-2]
            concept_id = splitted_line[-1]
            mention_splitted = mention.split('|')
            concept_id_splitted = concept_id.split()
            query_id = ann_file_path + '_{}'.format(row_id)
            if len(mention_splitted) > 1:
                for mention, concept_id in zip(mention_splitted, concept_id_splitted):
                    data.append({'entity_text': mention, 'label': concept_id, 'query_id': query_id, 'entity_id': row_id})
            else:
                data.append({'entity_text': mention, 'label': concept_id, 'query_id': query_id,  'entity_id': row_id})
    return data


def read_dataset(dataset_folder: str) -> List[Entity]:
    ann_file_pattern = os.path.join(dataset_folder, '*.concept')
    dataset = []
    for ann_file_path in glob(ann_file_pattern):
        dataset += read_annotation_file(ann_file_path)
    return dataset


def process_dataset(test_dataset: str, train_dataset: str) -> List[Entity]:
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)
    refined_set = test_df[~test_df.entity_text.isin(train_df.entity_text)]
    return refined_set.drop_duplicates().to_dict('records')


def save_dataset(dataset: List[Entity], path: str):
    if not os.path.exists(path): os.mkdir(path)
    fpath = os.path.join(path, '0.concept')
    with open(fpath, 'w', encoding='utf-8') as output_stream:
        for entity in dataset:
            entity_id = -1
            if 'entity_id' in entity:
                entity_id = entity['entity_id']
            output_stream.write(f"{entity_id}||0|0||Disease||{entity['entity_text']}||{entity['label']}\n")


def read_clinical_trials_dataset(fpath: str, single_concept: bool = False) -> List[Entity]:
    dataset_df = pd.read_csv(fpath, sep='\t', encoding='utf-8').drop_duplicates()
    if 'drugbank_ids' in dataset_df.columns:
        concept_id_column = 'drugbank_ids'
    else:
        concept_id_column = 'indication_ids'
    dataset_df = dataset_df[~dataset_df[concept_id_column].isnull()]
    dataset_df = dataset_df[~dataset_df.name.isnull()]
    dataset_df['name'] = dataset_df.name.str.lower()
    dataset_df = dataset_df[dataset_df[concept_id_column].str.strip() != '[]']
    dataset_df[concept_id_column] = dataset_df[concept_id_column].apply(lambda t: t.replace('[', '').replace(']', '').replace("'", ''))
    dataset_df.rename(columns={'name': 'entity_text', concept_id_column: 'label'}, inplace=True)
    dataset_df['query_id'] = range(dataset_df.shape[0])
    dataset_df['label'] = dataset_df['label'].apply(lambda t: t.split(',') if t else [])
    if single_concept or True:
        dataset_df = dataset_df[dataset_df.label.apply(len) == 1]
    dataset_df = dataset_df.explode('label')
    return dataset_df.to_dict('records')

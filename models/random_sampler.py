import numpy as np
import pandas as pd

from typing import List, Any


def is_equal(label_1: str, label_2: str) -> bool:
    """
    Comparing composite concept_ids
    """
    return len(set(label_1.replace('+', '|').split("|")).
               intersection(set(label_2.replace('+', '|').split("|")))) > 0


class RandomSampler:
    def __init__(self, vocab_path: str, search_count: int) -> None:
        self.vocab = self.load_vocab(vocab_path)
        self.search_count = search_count

    @staticmethod
    def load_vocab(vocab_path: str) -> pd.DataFrame:
        vocab = []
        concept_ids = []
        with open(vocab_path, encoding='utf-8') as input_stream:
            for line in input_stream:
                vocab.append(line.strip().split('||')[1])
                concept_ids.append(line.split('||')[0])
        return pd.DataFrame({'concept_name': vocab, 'concept_id': concept_ids})

    def get_candidates(self, labels: List[str]) -> List[Any]:
        labels_df = pd.DataFrame({'concept_id': labels})
        labels_df['order'] = range(labels_df.shape[0])
        positive_examples = pd.merge(labels_df, self.vocab, on='concept_id')
        negative_examples = pd.DataFrame({'order': [], 'concept_id': [], 'concept_name': []})
        for order, label in enumerate(labels):
            rand_order = np.random.choice(self.vocab.shape[0], size=self.search_count, replace=False)
            negatives_examples_for_label = self.vocab.iloc[rand_order][self.vocab.concept_id != label]
            negatives_examples_for_label['order'] = order
            negative_examples = pd.concat([negative_examples, negatives_examples_for_label])
        candidates = pd.concat([positive_examples, negative_examples])
        candidates['distances'] = 0.0
        concept_ids = candidates.groupby('order')['concept_id'].apply(lambda t: list(t)).reset_index(). \
            sort_values('order').drop('order', axis=1)
        distances = candidates.groupby('order')['distances'].apply(lambda t: list(t)).reset_index(). \
            sort_values('order').drop('order', axis=1)
        concept_names = candidates.groupby('order')['concept_name'].apply(lambda t: list(t)).reset_index(). \
            sort_values('order').drop('order', axis=1)
        predicted_labels = pd.concat([concept_ids, distances, concept_names], axis=1)
        return predicted_labels.values.tolist()

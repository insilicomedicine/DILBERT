import numpy as np
import faiss
import torch
import os
import logging
from preprocess import preprocess, combine_candidates, split_entity
from sentence_transformers import models, SentenceTransformer
from sentence_transformers import SentencesDataset
from sentence_transformers.evaluation import SentenceEvaluator, TripletEvaluator
from torch.utils.data import DataLoader
from sentence_transformers.losses import TripletLoss
from sklearn.model_selection import train_test_split

from sentence_transformers.readers import TripletReader
from typing import List, Tuple, Any, Optional, Dict
from tqdm import tqdm

logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s')


class RankingMapper:
    def __init__(self, model_path: str, vocab_path: Optional[str] = None, threshold: float = 55,
                 search_count: int = 10, gpu: bool = True) -> None:
        logging.info("Loading Biobert")
        self.model = self.load_model(model_path)
        if vocab_path is not None:
            logging.info("Loading vocab")
            self.vocab, self.concept_ids_vocab = self.load_vocab(vocab_path)
            logging.info(f"Loaded vocab contains {len(self.vocab)} concept names")
            self.concept2idx = self.get_label2idx_mapping(self.concept_ids_vocab)
        self.gpu = gpu
        self.index = None
        self.cpu_index = None
        self.search_count = search_count
        self.threshold = threshold

    def init_embeddings(self) -> None:
        logging.info("Calculating embeddings")
        concept_embeddings = self.encode(self.vocab)
        logging.info("Creating cpu index")
        self.cpu_index = faiss.IndexFlatL2(768)
        if self.gpu:
            logging.info("Moving Index to GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.cpu_index)
        else:
            self.index = self.cpu_index
        logging.info("Adding embeddings to index")
        self.index.add(concept_embeddings)

        # sanity check
        logging.info("Sanity check")
        logging.info(len(self.vocab), len(concept_embeddings))
        assert len(self.vocab) == len(concept_embeddings)
        logging.info("Done loading ranking model")

    @staticmethod
    def load_model(path: str) -> SentenceTransformer:
        checkpoint_files = os.listdir(path)
        if 'pytorch_model.bin' in checkpoint_files:
            word_embedding_model = models.Transformer(path)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)
            return SentenceTransformer(modules=[word_embedding_model, pooling_model])
        return SentenceTransformer(path)

    def encode(self, texts: List[str]) -> np.array:
        batch_size = 4096
        embeddings = []
        print(texts)
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(len(texts), batch_start + batch_size)
            batch_embeddings = self.model.encode(texts[batch_start:batch_end], show_progress_bar=False, batch_size=512)
            embeddings.append(batch_embeddings)
        embeddings = np.concatenate(embeddings)
        return np.vstack(embeddings)

    @staticmethod
    def load_vocab(vocab_path: str) -> Tuple[np.array, np.array]:
        vocab = []
        concept_ids = []
        with open(vocab_path, encoding='utf-8') as input_stream:
            for line_id, line in enumerate(input_stream):
                vocab.append(line.strip().split('||')[1])
                concept_ids.append(line.split('||')[0])
        return np.array(vocab), np.array(concept_ids)

    def get_candidates(self, entities: List[str]) -> List[Any]:
        if self.index is None:
            self.init_embeddings()
        batch_size = 256
        embeddings_all = self.encode(entities)
        logging.info("Embedded entities")
        predicted_labes = []
        for batch_start_idx in range(0, len(entities), batch_size):
            batch_end_idx = min(batch_start_idx + batch_size, len(entities))
            embeddings = embeddings_all[batch_start_idx:batch_end_idx]
            batch_distances, batch_ids = self.index.search(embeddings, self.search_count)
            for distances, ids in zip(batch_distances, batch_ids):
                top_ids = np.argsort(distances)
                top_concept_idx = ids[top_ids]
                top_concept_id = self.concept_ids_vocab[top_concept_idx.tolist()]
                top_distances = distances[top_ids]
                top_concept_names = self.vocab[top_concept_idx.tolist()]
                predicted_labes.append([top_concept_id.tolist(), top_distances.tolist(), top_concept_names.tolist()])
        return predicted_labes

    def predict(self, entities: List[Dict[str, Any]],
                split: bool = False, pp: bool = False, out_of_kb: bool = False) -> List[Any]:
        if split:
            entities_pp = [sp_entity for entity in entities for sp_entity in split_entity(entity)]
        else:
            entities_pp = entities
        if pp:
            entities_pp = preprocess(entities_pp)
        entity_texts = [entity['entity_text'] for entity in entities_pp]
        candidates = self.get_candidates(entity_texts)
        for entity, candidate_list in zip(entities_pp, candidates):
            entity['label'] = candidate_list
        if pp:
            entities_pp = combine_candidates(entities_pp)
        for entity in tqdm(entities_pp):
            distances = entity['label'][1]
            concept_ids = entity['label'][0]
            pred_labels = []
            for dist, concept_id in zip(distances, concept_ids):
                if dist > 55 and out_of_kb: break
                pred_labels.append(concept_id)
            pred_labels.append('NIL')
            entity['label'] = pred_labels

        return entities

    def train(self, data_reader: TripletReader, batch_size: int, epochs: int,
              output_dir: str, triplets_file: str, test_size: float = 0.3) -> None:
        triplets = data_reader.get_examples(triplets_file)
        train_triplets, test_triplets = train_test_split(triplets, test_size=test_size)
        train_data = SentencesDataset(train_triplets, model=self.model)
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=1)

        train_loss = TripletLoss(model=self.model)
        warmup_steps = int(len(train_triplets) / batch_size * epochs * 0.1)
        torch.multiprocessing.set_start_method('spawn')
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       evaluator=TripletEvaluator.from_input_examples(test_triplets),
                       epochs=epochs,
                       evaluation_steps=int(len(train_triplets) / batch_size),
                       warmup_steps=warmup_steps,
                       output_path=output_dir
                       )

    @staticmethod
    def get_label2idx_mapping(concept_ids: np.ndarray) -> Dict[str, int]:
        label2idx = {}
        for concept_id in concept_ids:
            if concept_id not in label2idx:
                label2idx[concept_id] = len(label2idx)
        return label2idx

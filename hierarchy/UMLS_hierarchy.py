import pandas as pd
from typing import Dict
from tqdm import tqdm
import ast

class UMLSGraph:
    def __init__(self, umls_path: str) -> None:
        self.umls = pd.read_csv(umls_path)
        self.tree = {}
        self.create_tree(self.umls)

    def create_tree(self, mesh: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        for row_idx, UMLS_ENTRY in tqdm(enumerate(mesh.itertuples(index=False)), total=mesh.shape[0]):
            node = self.get_node(UMLS_ENTRY)
            if node.REL=='CHD':
                node['parents'].extend(ast.literal_eval(node.SecondAUI))
            if node.REL=='PAR':
                node['children'].extend(ast.literal_eval(node.SecondAUI))

    def get_node(self, row):
        if row.FirstCUI in self.tree:
            node = self.tree[row.FirstAUI]
        else:
            node = {
                'concept_id': row.FirstAUI,
                'concept_name': row.NAME,
                'parents': [],
                'children': []
            }
            self.tree[row.FirstCUI] = node
        return node

    def get_parents(self, concept_id: str):
        if concept_id not in self.tree: return []
        return self.tree[concept_id]['parents']

    def get_children(self, concept_id: str):
        if concept_id not in self.tree: return []
        return self.tree[concept_id]['children']

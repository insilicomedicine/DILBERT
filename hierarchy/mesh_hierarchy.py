import pandas as pd
from typing import Dict
from tqdm import tqdm


class MeSHGraph:
    def __init__(self, mesh_path: str) -> None:
        self.mesh = pd.read_csv(mesh_path)
        self.mesh['parsedTNL'] = self.mesh['TreeNumberList'].str.replace(r'\[|\]', '').str.split(r', ?')
        self.mesh = self.mesh.explode('parsedTNL')
        self.mesh = self.mesh[self.mesh.parsedTNL.apply(len) != 0]
        self.mesh = self.mesh.sort_values('parsedTNL')
        self.tree = {}
        self.create_tree(self.mesh)

    def create_tree(self, mesh: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        for row_idx, mesh_entry in tqdm(enumerate(mesh.itertuples(index=False)), total=mesh.shape[0]):
            node = self.get_node(mesh_entry)
            TNL = mesh_entry.parsedTNL
            for row_idx_s, mesh_entry_s in mesh[row_idx + 1:].iterrows():
                node_s = self.get_node(mesh_entry_s)
                TNL_s = mesh_entry_s.parsedTNL
                if TNL_s.startswith(TNL):
                    node['children'].append(node_s['concept_id'])
                    node_s['parents'].append(node['concept_id'])
                else:
                    break

    def get_node(self, row):
        if row.DescriptorUI in self.tree:
            node = self.tree[row.DescriptorUI]
        else:
            node = {
                'concept_id': row.DescriptorUI,
                'concept_name': row.DescriptorName,
                'parents': [],
                'children': []
            }
            self.tree[row.DescriptorUI] = node
        return node

    def get_parents(self, concept_id: str):
        if concept_id not in self.tree: return []
        return self.tree[concept_id]['parents']

    def get_children(self, concept_id: str):
        if concept_id not in self.tree: return []
        return self.tree[concept_id]['children']

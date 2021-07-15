import re
from typing import Dict, List, Any


DELIMITERS = [' and ', ' or ', ', ', '/', ' plus ', ' vs ', 'combined with',
              'in combination with', 'admixed with', 'matching']

IRRELEVANT = ['[0-9]*\.?[0-9]+ ?%', '[0-9]*\.?[0-9]+ ?mg', '[0-9]*\.?[0-9]+ ?mcg', '[0-9]*\.?[0-9]+ ?ug',
              '[0-9]*\.?[0-9]+ ?mg/kg', '[0-9]*\.?[0-9]+ ?g', '[0-9]*\.?[0-9]+ ?mg/ml', '[0-9]*\.?[0-9]+ ?mg/day',
              '[0-9]*\.?[0-9]+ ?mg/d', '[0-9]*\.?[0-9]+ ?u', '[0-9]*\.?[0-9] ?iu/ml', '[0-9]*\.?[0-9]+ ?u/ml',
              '[0-9]*\.?[0-9]+ ?u/day', '[0-9]*\.?[0-9]+ ?u/d', '[0-9]*\.?[0-9]+ ?iu', '.*\:', 'injection',
              'inhalation', 'infusion', 'surgery', 'treatment', 'solution', 'suspension',
              'implant', ' tablet', 'tablets', ' tab', 'capsule', 'caplet', 'patch',
              'cream', ' gel', 'ointment', 'injector', 'syringe', 'eye drops', 'film',
              'aerosol', 'women', 'men', 'humans', 'aged', 'daily', 'weekly', 'monthly',
              'once', 'twice', 'dose', 'administered at', 'topical', 'transdermal',
              'ophthalmic', 'sublingual', 'oral', 'intravenous', 'subcutaneous',
              'intranasal', 'nasal', 'intrapleural', 'metered-dose inhaler (MDI)',
              'breath-actuated inhaler (BAI)', 'investigational medicinal product (IMP)'
              'coated', 'modified', 'fast acting', 'fast-acting', 'active', 'long acting',
              'long-acting', 'delayed-release', 'delayed release', 'extended-release',
              'extended release', ', ?[0-9]*\.?[0-9]+', '^-', '^’s ?', '^\)', '^\)-', '\($', '/$',
              ' (?=`s )', " (?='s )", '-a$', " '(?= )", "^'s ?", "^`s ?", ' (?=’s )', '^/']

Entity = Dict[str, Any]


def resolve_parenthesis(entity: Entity) -> List[Entity]:
    text = entity['entity_text']
    if '(' in text and ')' in text and text.index('(') < text.index(')'):
        inside_text = re.findall('\(.*\)', text)[0][1:-1]
        external_text = re.sub('\(.*\)', '', text)
        return [{'entity_id': entity['entity_id'], 'entity_text': inside_text},
                {'entity_id': entity['entity_id'], 'entity_text': external_text}]
    elif '[' in text and ']' in text and text.index('[') < text.index(']'):
        inside_text = re.findall('\[.*\]', text)[0][1:-1]
        external_text = re.sub('\[.*\]', '', text)
        return [{'entity_id': entity['entity_id'], 'entity_text': inside_text},
                {'entity_id': entity['entity_id'], 'entity_text': external_text}]
    else:
        return [entity]


def remove_irrelevant(entity: Entity) -> Entity:
    entity['entity_text'] = re.sub('|'.join(IRRELEVANT), '', entity['entity_text'].lower())
    return entity


def preprocess(entities: List[Entity], parenthesis: bool = True, irrelevant: bool = True) -> List[Entity]:
    processed_entities = entities
    if irrelevant:
        processed_entities = [remove_irrelevant(entity) for entity in processed_entities]
    if parenthesis:
        processed_entities = [pp_entity for entity in processed_entities for pp_entity in resolve_parenthesis(entity)]
    return processed_entities


def split_entity(entity: Entity) -> List[Entity]:
    splitted_entities = []
    text = entity['entity_text']
    entity_id = entity['entity_id']
    splitted_texts = re.split('|'.join(DELIMITERS), text)
    for split_id, splitted_text in enumerate(splitted_texts):
        splitted_entities.append({'entity_id': f'{entity_id}_{split_id}',
                                  'entity_text': splitted_text, 'label': entity['label']})
    return splitted_entities


def combine_candidates(entities: List[Entity]) -> List[Entity]:
    merged_entities = []
    prev_entity = None
    entities = sorted(entities, key=lambda entity: entity['entity_id'])
    for entity in entities:
        if prev_entity is not None and entity['entity_id'] == prev_entity['entity_id']:
            candidates = list(zip(*entity['label'])) + list(zip(*prev_entity['label']))
            candidates = sorted(candidates, key=lambda t: t[1])
            prev_entity['label'] = list(zip(*candidates))
            print('merged')
        else:
            merged_entities.append(entity)
        prev_entity = entity
    return merged_entities

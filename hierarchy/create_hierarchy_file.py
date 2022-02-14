from argparse import ArgumentParser
import pandas as pd
import json


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--save_to')
    return parser.parse_args()


def parse_record(record_lines):
    entry = {'Terms': [], 'TreeNumberList': [], 'ScopeNotes': []}
    for record_line in record_lines:
        #DescriptorName
        if record_line.startswith('MH ='):
            entry['DescriptorName'] = record_line.replace('MH = ', '').strip()
        #DescriptorUI
        elif record_line.startswith('UI ='):
            entry['DescriptorUI'] = record_line.replace('UI = ', '').strip()
        #Terms
        elif record_line.startswith('ENTRY ='):
            term = record_line.replace('ENTRY = ', '').strip().split('|')[0].strip()
            entry['Terms'].append(term)
        #TreeNumberList
        elif record_line.startswith('MN ='):
            entry['TreeNumberList'].append(record_line.replace('MN = ', '').strip())
        #ScopeNotes
        elif record_line.startswith('MS ='):
            entry['ScopeNotes'].append(record_line.replace('MS = ', '').strip())
    return entry


def main():
    args = get_args()
    entries = []
    with open(args.input_file, encoding='utf-8') as input_stream:
        record_lines = []
        for line in input_stream:
            if line == '*NEWRECORD\n' and len(record_lines) > 0:
                entries.append(parse_record(record_lines))
                record_lines = []
            else:
                record_lines.append(line)
    entries_df = pd.DataFrame(entries)
    entries_df['Terms'] = entries_df['Terms'].apply(json.dumps).str.replace('"', '')
    entries_df['TreeNumberList'] = entries_df['TreeNumberList'].apply(json.dumps).str.replace('"', '')
    entries_df['ScopeNotes'] = entries_df['ScopeNotes'].apply(json.dumps).str.replace('"', '')
    entries_df[['DescriptorName','DescriptorUI','Terms','TreeNumberList','ScopeNotes']].to_csv(args.save_to, index=False, encoding='utf-8')


if __name__ == '__main__':
    main()

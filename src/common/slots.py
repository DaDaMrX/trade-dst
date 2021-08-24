import json
import os


def load_slots(data_dir='data', enforce_refresh=False):
    path = os.path.join(data_dir, 'cache', 'slots.txt')
    if not enforce_refresh and os.path.exists(path):
        with open(path) as f:
            slots = f.read().strip().splitlines()
        return slots

    exp_domains = ['hotel', 'train', 'restaurant', 'attraction', 'taxi']

    ontology_path = os.path.join(data_dir, 'multi-woz/MULTIWOZ2 2/ontology.json')
    with open(ontology_path) as f:
        ontology = json.load(f)
    
    ontology_domains = {k: v for k, v in ontology.items() if k.split('-')[0] in exp_domains}
    slots = [k.replace(' ', '').lower() if ('book' not in k) else k.lower() for k in ontology_domains.keys()]

    dump_dir = os.path.join(data_dir, 'cache')
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    dump_path = os.path.join(dump_dir, 'slots.txt')
    with open(dump_path, 'w') as f:
        f.write('\n'.join(slots) + '\n')

    return slots

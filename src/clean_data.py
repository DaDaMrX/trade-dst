import json
import os
import sys
import re

GENERAL_TYPO = {
    # type
    "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
    "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
    "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
    "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
    # area
    "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
    "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
    "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
    "centre of town":"centre", "cb30aq": "none",
    # price
    "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
    # day
    "next friday":"friday", "monda": "monday", 
    # parking
    "free parking":"free",
    # internet
    "free internet":"yes",
    # star
    "4 star":"4", "4 stars":"4", "0 star rarting":"none",
    # others 
    "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
    '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",  
}


def fix_value(slot, value):
    # general typos
    if value in GENERAL_TYPO.keys():
        value = value.replace(value, GENERAL_TYPO[value])
    
    # miss match slot and value 
    if  slot == "hotel-type" and value in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
        slot == "hotel-internet" and value == "4" or \
        slot == "hotel-pricerange" and value == "2" or \
        slot == "attraction-type" and value in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
        "area" in slot and value in ["moderate"] or \
        "day" in slot and value == "t":
        value = "none"
    elif slot == "hotel-type" and value in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
        value = "hotel"
    elif slot == "hotel-star" and value == "3 star hotel":
        value = "3"
    elif "area" in slot:
        if value == "no": value = "north"
        elif value == "we": value = "west"
        elif value == "cent": value = "centre"
    elif "day" in slot:
        if value == "we": value = "wednesday"
        elif value == "no": value = "none"
    elif "price" in slot and value == "ch":
        value = "cheap"
    elif "internet" in slot and value == "free":
        value = "yes"

    # some out-of-define classification slot values
    if  slot == "restaurant-area" and value in ["stansted airport", "cambridge", "silver street"] or \
        slot == "attraction-area" and value in ["norwich", "ely", "museum", "same area as hotel"]:
        value = "none"

    return value


def fix_book_number(sent):
    sent = re.sub(
        r'\breference number\b.*?\b\w{8}\b',
        r'reference number is REFNUM',
        sent,
    )
    sent = re.sub(
        r'\breference code is \w{8}\b',
        r'reference code is REFNUM',
        sent,
    )
    sent = re.sub(
        r'\breference is \w{8}\b',
        r'reference is REFNUM',
        sent,
    )
    sent = re.sub(
        r'\breference # \w{8}\b',
        r'reference # REFNUM',
        sent,
    )
    sent = re.sub(
        r'\bref # \w{8}\b',
        r'ref # REFNUM',
        sent,
    )
    sent = re.sub(
        r'\breference \w{8}\b',
        r'reference REFNUM',
        sent,
    )
    sent = re.sub(r'\btr\d{4}\b', 'TRAINNUM', sent)
    sent = re.sub(r'\b\d{11}\b', 'PHONENUM', sent)
    return sent


def clean_data(input_path, output_path, enforce_refresh=False):
    if os.path.exists(output_path) and not enforce_refresh:
        return

    with open(input_path) as f:
        data_input = json.load(f)

    # Clean: filter domain
    exp_domains = ['hotel', 'train', 'restaurant', 'attraction', 'taxi']
    data = []
    for dialog in data_input:
        if any(domain not in exp_domains for domain in dialog['domains']):
            continue
        else:
            data.append(dialog)

    # Clean
    data_output = []
    for dialog_input in data:
        dialog_output = {
            'dialogue_idx': None,
            'domains': None,
            'dialogue': None,
        }
        data_output.append(dialog_output)

        dialog_output['dialogue_idx'] = dialog_input['dialogue_idx']
        dialog_output['domains'] = sorted(dialog_input['domains'])  # Sort domains
        dialog_output['dialogue'] = []
        for i, turn_input in enumerate(dialog_input['dialogue']):
            assert i == turn_input['turn_idx']  # Insure turn indexes are incremental
            turn_output = {
                'turn_idx': None,
                'domain': None,
                'system_transcript': None,
                'transcript': None,
                'belief_state': None,
            }
            dialog_output['dialogue'].append(turn_output)

            turn_output['turn_idx'] = turn_input['turn_idx']
            turn_output['domain'] = turn_input['domain']

            system_transcript = turn_input['system_transcript'].strip()
            system_transcript = fix_book_number(system_transcript)  # Clean: fix book number
            transcript = turn_input['transcript'].strip()
            transcript = fix_book_number(transcript)  # Clean: fix book number

            turn_output['system_transcript'] = system_transcript
            turn_output['transcript'] = transcript
            turn_output['belief_state'] = []
            for state_input in turn_input['belief_state']:
                domain_slot = state_input['slots'][0][0]
                value = state_input['slots'][0][1].strip()
                value = fix_value(domain_slot, value)  # Clean: fix value typos
                if value == '' or value == 'none':  # Clean: remove value which is empty or none
                    continue
                state = f'{domain_slot}-{value}'
                turn_output['belief_state'].append(state)
            turn_output['belief_state'].sort()  # Sort slots

            assert None not in turn_output
        assert None not in dialog_output

    print(f'#Dialog Input: {len(data_input)}, Output: {len(data_output)}')

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(output_path, 'w') as f:
        json.dump(data_output, f, indent=4)


if __name__ == '__main__':
    assert len(sys.argv) == 3
    clean_data(
        input_path=sys.argv[1],
        output_path=sys.argv[2],
        enforce_refresh=True,
    )


'''
python src/clean_data.py data/dev_dials.json data/clean/valid.json
python src/clean_data.py data/train_dials.json data/clean/train.json
python src/clean_data.py data/test_dials.json data/clean/test.json
'''

def load_gate_dict():
    gate_dict = {'ptr': 0, 'dontcare': 1, 'none': 2}
    return gate_dict

def load_inv_gate_dict():
    gate_dict = load_gate_dict()
    inv_gate_dict = {v: k for k, v in gate_dict.items()}
    return inv_gate_dict

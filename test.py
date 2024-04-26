import numpy as np

def one_hot_encode_dna_sequence(sequence):

    base_dict = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    num_bases = len(base_dict)
    encoding = {}

    for i, base in enumerate(sequence):
        encoding[i, base_dict[base]] = 1

    return encoding


input_sequence = input("")
dna_sequences = input_sequence.split(',')

encoded_sequences = []
for seq in dna_sequences:
    encoded_seq = one_hot_encode_dna_sequence(seq)
    encoded_sequences.append(encoded_seq)


for i, seq in enumerate(encoded_sequences):
    print(f"DNA sequence {i+1} encoding:")
    print(seq)
    print()

# transcript DNA sequence
dna = 'ATGCCATAG'
# translate the DNA sequence into a protein sequence
protein = dna.translate()
# print the protein sequence

# a def to transcript DNA to RNA
def dna_to_rna(dna):
    # replace the T with U
    rna = dna.replace('T', 'U')
    return rna

# Main CLI (Command Line Interface) script 
"""
The current script accepts gene names** (e.g., TP53) as input. 
The workflow is:

        python -m src.cli --gene <GENE_NAME> --output <FILENAME.fa>

To test it, you can try this example:
        python -m src.cli --gene TP53 --output tp53_sequence.fa

Example valid inputs:
- gene TP53→ Fetches the TP53 gene sequence.
- gene BRCA1 → Fetches the BRCA1 gene sequence.
- gene Tp53 → Fetched the TP53 gene seq since the code handles lower case entries.

Example invalid inputs:
- gene XYZ123 → Fails (gene does not exist).

"""

import argparse
from .utils import fetch_gene_sequence, validate_dna_sequence, validate_sequence_length

def main():
    """
     Detailed explanation: 

    Lookup Gene ID (Query Ensembl’s REST API to convert TP53 → Ensembl ID (e.g., ENSG00000141510).)
    Fetch Sequence (Retrieves the DNA sequence for the Ensembl ID.)
    Validate Sequence (Checks for invalid characters (e.g., X, N) and minimum length (23 bp).)
    """
    parser = argparse.ArgumentParser(description="Fetch and validate gene sequences for CRISPR design.")
    parser.add_argument("--gene", type=str, help="Gene name (e.g., TP53)")
    parser.add_argument("--output", type=str, help="Output file to save the sequence")
    args = parser.parse_args()
    
    try:
        # Step 1: Fetch sequence
        sequence = fetch_gene_sequence(args.gene)
        
        # Step 2: Validate (redundant but explicit)
        if not validate_dna_sequence(sequence):
            raise ValueError("Invalid DNA characters detected.")
        if not validate_sequence_length(sequence):
            raise ValueError("Sequence too short.")
        
        # Step 3: Save/Write the validated sequence to a file (for e.g., tp53_sequence.fa).
        if args.output:
            with open(args.output, "w") as f:
                f.write(sequence)
        print(f"Success! Sequence for {args.gene}:\n{sequence}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


# Main CLI (Command Line Interface) script
"""
The current script accepts gene names** (e.g., TP53) as input. 
The workflow is:

        python -m src.cli --gene <GENE_NAME> --pam NGG --output <FILENAME.fa>

To test it, you can try this example:
        python -m src.cli --gene TP53 --pam NGG --output tp53_sgrnas.csv

Example valid inputs:
- gene TP53→ Fetches the TP53 gene sequence.
- gene BRCA1 → Fetches the BRCA1 gene sequence.
- gene Tp53 → Fetched the TP53 gene seq since the code handles lower case entries.

Example invalid inputs:
- gene XYZ123 → Fails (gene does not exist).

"""

import argparse
import pandas as pd
import numpy as np

from .utils import (
    fetch_gene_sequence,
    validate_dna_sequence,
    validate_sequence_length
)
from .pam import find_pam_sites, filter_sgrnas
from .scoring import DeepCRISPRScorer

def compute_gc(seq):
    """
    Compute the GC content of a DNA sequence.
    
    Args:
        seq (str): DNA sequence.
    Returns:
        float: GC content as a fraction.
    """
    gc_count = sum(1 for base in seq if base in "GC")
    return gc_count / len(seq)

def off_target_analysis(sgRNA):
    """
    Placeholder for off-target analysis.
    
    In a full implementation, this function would search the human genome
    (e.g., hg38) for similar sequences using a tool like Bowtie or BLAST and
    return the number of mismatches for the most similar off-target.
    
    For now, it returns "NA".
    
    Args:
        sgRNA (str): sgRNA sequence.
    Returns:
        str: Off-target mismatches info.
    """
    # TODO: Integrate off-target search using a human genome reference.
    return "NA"

def main():
    """
    Command-Line Interface for CRISPR sgRNA design.

    Example usage:
        python -m src.cli --gene TP53 --pam NGG --model_dir models/ --output tp53_sgrnas.csv

    Steps:
      1. Fetch gene sequence from Ensembl (if --gene is provided).
      2. Find PAM sites (both strands).
      3. Filter sgRNAs (GC content, homopolymers, etc.).
      4. Extract 30bp sequences (5bp up/downstream).
      5. Load DeepCRISPR model & predict efficiency scores.
      6. Compile results including GC content and off-target mismatch info.
      7. Save the output. 
    """

    parser = argparse.ArgumentParser(description="CRISPR sgRNA Design Tool")
    parser.add_argument("--gene", type=str, help="Gene name (e.g., TP53)")
    parser.add_argument("--pam", type=str, default="NGG", help="PAM motif (default: NGG)")
    parser.add_argument("--model_dir", type=str, default="models/", 
                        help="Path to DeepCRISPR checkpoint directory")
    parser.add_argument("--output", type=str, help="Output CSV file")
    args = parser.parse_args()

    try:
        # 1. Fetch sequence (for human gene using Ensembl REST API)
        if not args.gene:
            raise ValueError("Please provide a --gene name (e.g., TP53).")
        sequence = fetch_gene_sequence(args.gene)
        if not validate_dna_sequence(sequence):
            raise ValueError("Fetched sequence contains invalid characters.")
        if not validate_sequence_length(sequence):
            raise ValueError("Fetched sequence is too short (< 23 bp).")

        # 2. Identify PAM sites (search both strands)
        candidates = find_pam_sites(sequence, pam=args.pam, strand="both")
        filtered = filter_sgrnas(candidates)  # Filter based on design rules
        
        # 3. Extract 30bp context (5bp upstream + 20bp sgRNA + 5bp downstream)
        sgRNAs_30bp = []
        valid_sgrnas = []
        for sgrna, start, end, strand in filtered:
            # Check if sgRNA length is exactly 20bp
            if end - start != 20:
                print(f"Skipping {sgrna}: Invalid length {end - start}bp (expected 20)")
                continue
            flank_start = start - 5  # 5bp upstream
            flank_end = end + 5      # 5bp downstream
            # Check bounds
            if flank_start < 0 or flank_end > len(sequence):
                print(f"Skipping {sgrna}: Flanking sequence out of bounds")
                continue
            flanking_sequence = sequence[flank_start:flank_end]
            if len(flanking_sequence) != 30:
                print(f"Skipping {sgrna}: Flanking seq is {len(flanking_sequence)}bp (expected 30)")
                continue
            sgRNAs_30bp.append(flanking_sequence)
            valid_sgrnas.append((sgrna, start, end, strand))
        
        if not valid_sgrnas:
            raise ValueError("No valid sgRNAs found after filtering/flanking checks.")
        
        # 4. Initialize DeepCRISPR scorer and predict efficiency.
        scorer = DeepCRISPRScorer(checkpoint_dir=args.model_dir)
        # One-hot encode the 30bp sequences and predict efficiency.
        encoded = np.array([scorer.one_hot_encode(seq) for seq in sgRNAs_30bp])
        preds = scorer.model.predict(encoded)  # shape => (num_sgRNAs, 1)
        efficiency_scores = preds.flatten().tolist()

        # 5. Compile results, now including GC content and off-target mismatches.
        results = []
        for (sgrna, start, end, strand), score in zip(valid_sgrnas, efficiency_scores):
            gc = compute_gc(sgrna)  # Compute GC content of the 20bp guide.
            mismatches = off_target_analysis(sgrna)  # Placeholder for off-target info.
            results.append({
                "sgRNA_20bp": sgrna,
                "Start": start,
                "End": end,
                "Strand": strand,
                "Efficiency": score,
                "GC_Content": gc,
                "OffTarget_Mismatches": mismatches
            })

        df = pd.DataFrame(results).sort_values("Efficiency", ascending=False)
        # Save to CSV if output is specified.
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved results to {args.output}")
        
        # Print top sgRNA information
        top = df.iloc[0]
        print(f"Top sgRNA: {top['sgRNA_20bp']} (Score: {top['Efficiency']:.4f}, GC: {top['GC_Content']:.2f}, OffTarget: {top['OffTarget_Mismatches']})")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
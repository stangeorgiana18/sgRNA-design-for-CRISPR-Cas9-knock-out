# Main CLI (Command Line Interface) script

"""
hg38 download (primary assembly):
       wget https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
For bowtie indexing, run the following command (it took ~8h on my computer for the genome to be indexed and ~1h to download), for reference):
       bowtie-build Homo_sapiens.GRCh38.dna.primary_assembly.fa GRCh38_primary


The current script accepts gene names** (e.g., TP53) as input. 
The workflow is:

        python -m src.cli --gene <GENE_NAME> --pam NGG --output <FILENAME.fa>

To test it, you can try this example:
        python -m src.cli --gene TP53 --pam NGG --model_dir src/models --bowtie_index bowtie_index/GRCh38_primary --output tp53_sgrnas.csv


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

def off_target_analysis(sgRNA, bowtie_index):
    """
    Run Bowtie to search for off-targets for the sgRNA in the human genome.
    
    This function writes the sgRNA (20bp guide) in FASTA format to a temporary file,
    then calls Bowtie using the provided Bowtie index (for the human genome, e.g., GRCh38_primary).
    
    It allows up to 2 mismatches (-v 2) and reports all valid alignments (-a).
    
    Args:
        sgRNA (str): 20bp sgRNA sequence.
        bowtie_index (str): Path to the Bowtie index prefix for the human genome.
                            For your index files, this should be "GRCh38_primary" (or with the directory path, e.g., "genome/GRCh38_primary").
    Returns:
        int: Number of off-target hits (i.e. alignments found by Bowtie).
    """
    import tempfile
    import subprocess
    import os
    
    # Write sgRNA to a temporary FASTA file.
    fasta_content = f">candidate\n{sgRNA}\n"
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".fa") as temp_fasta:
        temp_fasta.write(fasta_content)
        temp_fasta_path = temp_fasta.name

    # Build Bowtie command.
    # -v 2 : allow up to 2 mismatches.
    # -a    : report all valid alignments.
    # --best --strata : report the best alignments.
    # -f   : input is in FASTA format.
    cmd = [
        "bowtie", 
        "-v", "2", 
        "-a", 
        "--best", 
        "--strata", 
        bowtie_index,  # This should match the index prefix (e.g., "GRCh38_primary" or "genome/GRCh38_primary")
        "-f", 
        temp_fasta_path
    ]
    
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        # Count the number of alignment lines.
        off_target_hits = len(output.strip().splitlines())
    except subprocess.CalledProcessError as e:
        print("Bowtie error output:", e.output)
        off_target_hits = 0
    finally:
        os.remove(temp_fasta_path)
    
    return off_target_hits

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
      6. For each candidate, compute GC content and off-target hits using Bowtie.
      7. Save results to CSV.

      Example usage:
        python -m src.cli --gene TP53 --pam NGG --model_dir src/models --bowtie_index genome/hg38_index --output tp53_sgrnas.csv
    """

    parser = argparse.ArgumentParser(description="CRISPR sgRNA Design Tool")
    parser.add_argument("--gene", type=str, help="Gene name (e.g., TP53)")
    parser.add_argument("--pam", type=str, default="NGG", help="PAM motif (default: NGG)")
    parser.add_argument("--model_dir", type=str, default="models/", 
                        help="Path to DeepCRISPR checkpoint directory")
    parser.add_argument("--bowtie_index", type=str, help="Path to Bowtie index prefix for human genome (e.g., genome/hg38_index)")
    parser.add_argument("--output", type=str, help="Output CSV file")
    args = parser.parse_args()

    try:
        if not args.gene:
            raise ValueError("Please provide a --gene name (e.g., TP53).")
        
        # 1. Fetch gene sequence (human genome)
        sequence = fetch_gene_sequence(args.gene)
        if not validate_dna_sequence(sequence):
            raise ValueError("Fetched sequence contains invalid characters.")
        if not validate_sequence_length(sequence):
            raise ValueError("Fetched sequence is too short (< 23 bp).")
        
        # 2. Identify PAM sites (both strands)
        candidates = find_pam_sites(sequence, pam=args.pam, strand="both")
        filtered = filter_sgrnas(candidates)
        
        # 3. Extract 30bp context (5bp upstream + 20bp sgRNA + 5bp downstream)
        sgRNAs_30bp = []
        valid_sgrnas = []
        for sgrna, start, end, strand in filtered:
            if end - start != 20:
                print(f"Skipping {sgrna}: Invalid length {end - start}bp (expected 20)")
                continue
            flank_start = start - 5
            flank_end = end + 5
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
        encoded = np.array([scorer.one_hot_encode(seq) for seq in sgRNAs_30bp])
        preds = scorer.model.predict(encoded)
        efficiency_scores = preds.flatten().tolist()
        
        # 5. Compile results including GC content and off-target hits.
        results = []
        for (sgrna, start, end, strand), score in zip(valid_sgrnas, efficiency_scores):
            gc = compute_gc(sgrna)  # Compute GC content of the 20bp guide.
            # If bowtie_index is provided, run off-target analysis.
            if args.bowtie_index:
                off_target_hits = off_target_analysis(sgrna, args.bowtie_index)
            else:
                off_target_hits = "NA"
            results.append({
                "sgRNA_20bp": sgrna,
                "Start": start,
                "End": end,
                "Strand": strand,
                "Efficiency": score,
                "GC_Content": gc,
                "OffTarget_Hits": off_target_hits
            })
        
        df = pd.DataFrame(results).sort_values("Efficiency", ascending=False)
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved results to {args.output}")
        
        top = df.iloc[0]
        print(f"Top sgRNA: {top['sgRNA_20bp']} (Score: {top['Efficiency']:.4f}, GC: {top['GC_Content']:.2f}, OffTarget: {top['OffTarget_Hits']})")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
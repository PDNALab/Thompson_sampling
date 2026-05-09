#!/usr/bin/env python3
"""
strip_peptide_msa.py
--------------------
Remove peptide (Chain A) MSA rows and any paired MSA rows from a ColabFold
A3M file of a protein-peptide complex.

ColabFold A3M format for a complex (two chains, e.g. 25-AA peptide + 68-AA protein):

    # <len_A>,<len_B>          <- comment line with per-chain lengths
    \t                         <- empty paired-block header (tab-separated)
    >header_paired_1
    <seq_A><seq_B>             <- paired hit: concatenation of both chains
    ...
    >101                       <- unpaired block for chain A (index 101 = first chain)
    <seq_A only>
    ...
    >102                       <- unpaired block for chain B (index 102 = second chain)
    <seq_B only>
    ...

This script keeps:
  - The # comment line (updating it so the peptide length is still declared,
    but the peptide will have only its query sequence / a gap-only row).
  - A single query row for the peptide (the first sequence in chain A's block,
    i.e. the query itself) so ColabFold knows the chain still exists.
  - All unpaired MSA rows for chain B (the protein).
  - NO paired rows (rows that span both chains).
  - NO additional unpaired rows for chain A.

Usage
-----
    python strip_peptide_msa.py input.a3m output.a3m [--peptide-chain A]

Arguments
---------
    input.a3m        Path to the original ColabFold A3M file.
    output.a3m       Path to write the cleaned A3M file.
    --peptide-chain  Which chain to strip MSA for: A (default) or B.
"""

import argparse
import sys
import re


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_comment_lengths(line: str):
    """Parse the '# len1,len2' header line. Returns (len_a, len_b)."""
    # e.g.  "# 25,68"  or  "#25,68"
    m = re.match(r"#\s*(\d+)\s*,\s*(\d+)", line)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def read_a3m_blocks(path: str):
    """
    Read a ColabFold multi-chain A3M file and return:
        comment_line  : str  (the '# len_a,len_b' line)
        paired_seqs   : list of (header, seq) — paired block
        unpaired_a    : list of (header, seq) — chain-A-only block
        unpaired_b    : list of (header, seq) — chain-B-only block

    ColabFold uses a '\t' (tab character alone on a line) or a blank paired
    block to delimit sections, but the canonical marker is the special
    numeric headers '>101', '>102' (or '>1', '>2' in some versions).

    We detect blocks by those numeric-only headers.
    """

    with open(path, "r") as fh:
        raw = fh.read()

    lines = raw.splitlines()

    comment_line = ""
    paired_seqs: list = []
    unpaired_a:  list = []
    unpaired_b:  list = []

    # Identify the comment line
    i = 0
    while i < len(lines) and lines[i].startswith("#"):
        comment_line = lines[i]
        i += 1

    # Helper: collect (header, seq) pairs from a block of lines
    def collect_entries(block_lines):
        entries = []
        hdr = None
        seq_parts = []
        for ln in block_lines:
            if ln.startswith(">"):
                if hdr is not None:
                    entries.append((hdr, "".join(seq_parts)))
                hdr = ln
                seq_parts = []
            elif ln.strip() == "" or ln.strip() == "\t":
                continue
            else:
                seq_parts.append(ln.strip())
        if hdr is not None:
            entries.append((hdr, "".join(seq_parts)))
        return entries

    # Split remaining lines into three blocks using the sentinel headers
    # Sentinels: '>101' / '>1'  for chain A,  '>102' / '>2' for chain B
    # (ColabFold may use >101 and >102, or >1 and >2 depending on version)
    SENTINEL_A = re.compile(r"^>1{1,3}$")   # >1 or >101
    SENTINEL_B = re.compile(r"^>1{0,2}2$")  # >2 or >102

    rest = lines[i:]

    # Find sentinel positions
    idx_a = idx_b = None
    for j, ln in enumerate(rest):
        if SENTINEL_A.match(ln.strip()) and idx_a is None:
            idx_a = j
        elif SENTINEL_B.match(ln.strip()) and idx_b is None:
            idx_b = j

    if idx_a is None or idx_b is None:
        # Fallback: maybe only unpaired blocks exist (no paired section)
        # Try to find any ">1..." and ">2..." style markers
        for j, ln in enumerate(rest):
            if re.match(r"^>\d+$", ln.strip()):
                num = int(ln.strip()[1:])
                if num % 100 == 1 and idx_a is None:
                    idx_a = j
                elif num % 100 == 2 and idx_b is None:
                    idx_b = j

    if idx_a is not None and idx_b is not None:
        paired_block   = rest[:idx_a]
        unpaired_a_block = rest[idx_a:idx_b]
        unpaired_b_block = rest[idx_b:]
    elif idx_b is not None:
        paired_block   = []
        unpaired_a_block = []
        unpaired_b_block = rest[idx_b:]
    else:
        # Cannot detect structure; treat everything as paired
        print("WARNING: Could not detect chain block sentinels. "
              "Treating entire file as paired block.", file=sys.stderr)
        paired_block   = rest
        unpaired_a_block = []
        unpaired_b_block = []

    paired_seqs  = collect_entries(paired_block)
    unpaired_a   = collect_entries(unpaired_a_block)
    unpaired_b   = collect_entries(unpaired_b_block)

    return comment_line, paired_seqs, unpaired_a, unpaired_b


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def strip_peptide_msa(input_path: str, output_path: str, peptide_chain: str = "A"):
    """
    Read the ColabFold A3M, drop paired MSA rows and peptide unpaired rows,
    keep only the query sequence for the peptide, and write the result.
    """

    peptide_chain = peptide_chain.upper()
    if peptide_chain not in ("A", "B"):
        raise ValueError("--peptide-chain must be 'A' or 'B'.")

    comment_line, paired_seqs, unpaired_a, unpaired_b = read_a3m_blocks(input_path)

    len_a, len_b = parse_comment_lengths(comment_line)
    if len_a is None:
        raise ValueError(
            f"Could not parse chain lengths from comment line: '{comment_line}'\n"
            "Expected format: '# <len_chain_A>,<len_chain_B>'"
        )

    print(f"Parsed chain lengths  — Chain A (peptide): {len_a} AA, "
          f"Chain B (protein): {len_b} AA")
    print(f"Paired MSA rows       : {len(paired_seqs)}")
    print(f"Unpaired Chain-A rows : {len(unpaired_a)}")
    print(f"Unpaired Chain-B rows : {len(unpaired_b)}")

    # -----------------------------------------------------------------------
    # Determine which chain is the peptide and which is the protein
    # -----------------------------------------------------------------------
    if peptide_chain == "A":
        pep_len      = len_a
        prot_len     = len_b
        pep_unpaired  = unpaired_a
        prot_unpaired = unpaired_b
        pep_sentinel  = ">101"   # will be written as chain-A sentinel
        prot_sentinel = ">102"
    else:
        pep_len      = len_b
        prot_len     = len_a
        pep_unpaired  = unpaired_b
        prot_unpaired = unpaired_a
        pep_sentinel  = ">102"
        prot_sentinel = ">101"

    # -----------------------------------------------------------------------
    # Extract query sequence for the peptide (first entry in its unpaired block)
    # -----------------------------------------------------------------------
    if not pep_unpaired:
        # Fall back: extract from the first paired sequence
        if paired_seqs:
            first_paired_seq = paired_seqs[0][1]
            if peptide_chain == "A":
                pep_query_seq = first_paired_seq[:pep_len]
            else:
                pep_query_seq = first_paired_seq[pep_len:]
            pep_query_hdr = ">query"
            print("WARNING: No unpaired chain-A block found; "
                  "using query slice from first paired row.", file=sys.stderr)
        else:
            raise ValueError(
                "No unpaired peptide block and no paired block found. "
                "Cannot determine peptide query sequence."
            )
    else:
        pep_query_hdr, pep_query_seq = pep_unpaired[0]   # first = query itself

    # -----------------------------------------------------------------------
    # Build output
    # -----------------------------------------------------------------------
    out_lines = []

    # 1. Comment line (unchanged — lengths stay the same so ColabFold knows
    #    chain boundaries in the concatenated sequence)
    out_lines.append(comment_line)

    # 2. Paired block — EMPTY (we keep the paired sentinel but add nothing)
    #    ColabFold expects the paired block (even if empty) before the unpaired ones.
    #    We write a blank paired block (just no entries).
    out_lines.append("")   # blank line separating comment from paired block

    # 3. Chain-A unpaired sentinel  '>101'
    out_lines.append(">101")
    if peptide_chain == "A":
        # Only the query row for the peptide
        out_lines.append(pep_query_hdr)
        out_lines.append(pep_query_seq)
    else:
        # Protein is chain A — keep all protein rows
        for hdr, seq in prot_unpaired:
            out_lines.append(hdr)
            out_lines.append(seq)

    # 4. Chain-B unpaired sentinel  '>102'
    out_lines.append(">102")
    if peptide_chain == "B":
        # Only the query row for the peptide
        out_lines.append(pep_query_hdr)
        out_lines.append(pep_query_seq)
    else:
        # Protein is chain B — keep all protein rows
        for hdr, seq in prot_unpaired:
            out_lines.append(hdr)
            out_lines.append(seq)

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    with open(output_path, "w") as fh:
        fh.write("\n".join(out_lines) + "\n")

    print(f"\nDone. Written to: {output_path}")
    print(f"  Peptide (Chain {peptide_chain}) MSA rows kept : 1  (query only)")
    print(f"  Protein unpaired rows kept          : "
          f"{len(prot_unpaired)}")
    print(f"  Paired rows removed                 : {len(paired_seqs)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Strip peptide MSA and paired MSA rows from a ColabFold A3M file.\n"
            "Keeps only the query sequence for the peptide chain and all "
            "unpaired MSA rows for the protein chain."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input",  help="Input A3M file (from ColabFold MSA search)")
    parser.add_argument("output", help="Output A3M file (cleaned)")
    parser.add_argument(
        "--peptide-chain",
        default="A",
        choices=["A", "B"],
        help="Which chain is the peptide (default: A)",
    )

    args = parser.parse_args()
    strip_peptide_msa(args.input, args.output, args.peptide_chain)


if __name__ == "__main__":
    main()

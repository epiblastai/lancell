# Chromatin Accessibility Fragment Storage

ATAC-seq experiments produce millions of genomic fragments — short intervals
on a reference genome where the Tn5 transposase found open chromatin. Each
fragment is defined by three values: which chromosome it falls on, where it
starts, and how long it is. The challenge is storing these fragments so that
two very different access patterns are both fast:

1. **Per-cell retrieval**: "give me all fragments for cells X, Y, Z"
2. **Genomic region queries**: "give me all fragments in chr5:100000-200000"

No single sort order serves both. Cell-sorted data requires a full scan to
answer region queries; genome-sorted data requires a full scan to collect a
single cell's fragments. We store both orderings in the same zarr group.

## Storage layout

Each fragment is represented by three integers. Rather than storing them as
rows of a table, they are kept as three parallel flat arrays — element *i*
across all three arrays describes one fragment. This columnar layout
compresses well and enables vectorized reads.

```
{dataset}/chromatin_accessibility/
  cell_sorted/
    chromosomes    uint8
    starts         uint32
    lengths        uint16
  genome_sorted/
    cell_ids       uint32
    starts         uint32
    lengths        uint16
    chrom_offsets   int64
    end_max        uint32
```

The two orderings share identical fragment data; only the sort order and
auxiliary index arrays differ.

## Cell-sorted order

Fragments are sorted by (cell, chromosome, position). A CSR-style offsets
array of length `n_cells + 1` marks where each cell's fragments begin and
end in the flat arrays. Retrieving one cell's fragments is a single
contiguous slice — no filtering required.

Within each cell, fragments are ordered by genomic position. This makes the
`starts` array highly compressible with delta encoding: consecutive values
differ by small amounts, so the deltas pack into very few bits.

Because cell identity is implicit in the offsets array, no per-fragment
cell ID is stored. The `chromosomes` array (uint8, one byte per fragment)
records which chromosome each fragment belongs to.

## Genome-sorted order

Fragments are sorted by (chromosome, position) globally, interleaving
fragments from all cells. This ordering enables fast genomic range queries
but requires two additional structures:

**Chromosome boundary array** (`chrom_offsets`): A short array of length
`n_chromosomes + 1` recording where each chromosome's fragments begin and
end. This replaces the per-fragment chromosome array — chromosome identity
is implicit from position within the boundaries.

**Cell ID array** (`cell_ids`): Since fragments from different cells are
interleaved, each fragment carries an explicit cell index (uint32). This is
the main storage cost of the genome-sorted copy — roughly 4 bytes per
fragment before compression.

**Seek index** (`end_max`): An array storing the maximum end coordinate
(`start + length`) for every 128 consecutive fragments. This enables
O(log n) binary search to any genomic position without scanning the
chromosome.

### Why the seek index is necessary

Finding fragments that *start* within a query region is straightforward —
the starts array is sorted, so binary search works directly. The subtlety
is fragments that start *before* the query region but are long enough to
extend into it. Without additional information, there is no way to know how
far back to look.

The seek index solves this exactly. Each entry records the furthest-reaching
fragment end within a block of 128. Binary searching for the query start
coordinate finds the first block where any fragment could possibly reach
into the query region. All earlier blocks are guaranteed to contain no
overlapping fragments.

The block size of 128 matches the BP-128 bitpacking codec's internal block
size. Seeking to a block boundary means the codec can begin decoding at that
exact position without reading preceding blocks.

## Compression

The `starts` arrays in both orderings use delta encoding via a BP-128
bitpacking codec. Because fragments are sorted by position, consecutive
start coordinates differ by small amounts. These small deltas compress to
far fewer bits than the raw 32-bit coordinates — typically 10-15 bits per
element for dense genomic regions.

At chromosome boundaries in the genome-sorted array, one delta will be
large (the position resets). This causes a single 128-element block to
compress poorly, but with only ~24 chromosomes in a typical genome, the
impact is negligible.

The `cell_ids` array in genome-sorted order uses standard bitpacking without
delta encoding, since cell indices are effectively random within a genomic
region. Compression depends on the number of unique cells — fewer cells
means fewer bits per value.

## Access patterns

| Query | Ordering | Mechanism |
|-------|----------|-----------|
| All fragments for a cell | Cell-sorted | Slice `offsets[j]:offsets[j+1]` |
| All fragments for a set of cells | Cell-sorted | Multiple contiguous slices |
| All fragments in a genomic region | Genome-sorted | Binary search `end_max`, read + filter |
| Coverage pileup over a region | Genome-sorted | Region query, then bin and count |
| Peak/tile matrix construction | Genome-sorted | Stream sorted fragments against sorted peaks |

Region queries follow a two-step process: first narrow the read range using
the seek index, then filter the candidate fragments by overlap. The filter
keeps fragments where `start < query_end` and `start + length > query_start`.

## Relationship to BPCells

This design follows the BPCells paper's approach to ATAC-seq fragment
storage. The key ideas — genome-sorted order with a seek index, delta-encoded
start coordinates, chromosome boundary pointers — originate from that work.
The main adaptation is using zarr v3 sharded arrays with a registered
bitpacking codec instead of BPCells' custom file format.

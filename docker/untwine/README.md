# Patched untwine

The image rebuilds untwine 1.5.1 (hash-pinned release tarball) with these patches and
reports `untwine version (1.5.1+3dtrees.1)`; the image carries the label
`org.3dtrees.untwine.version`.

- `0001-reprocess-at-least-one-level-deeper.patch`: a first-pass cell of exactly
  100,000 points (`MaxPointsPerNode`) is reprocessed with zero extra levels, re-binned
  into its own `L-X-Y-Z.bin` and that file is then deleted, so the points the writer had
  already appended are lost while untwine exits 0. The patch splits such cells at least
  one level deeper. Found on dataset 1924 (66,430 and 49,401 points lost in two runs).
- `0002-report-3dtrees-patch-version.patch`: version suffix `+3dtrees.1`.

Drop both patches once an upstream untwine release contains the fix.

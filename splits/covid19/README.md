# COVID-19 Official20 V2 split identity

These manifests retain the existing sample order, dataset case identifiers, derived paths and geometry. Redundant GT statistics and raw absolute paths are omitted; no image or mask arrays are included. The adapter resolves paths starting with `data/` through DATA_ROOT. All train/validation/test identifiers originate from the public 20-case dataset.

PCG64 split seed 42 gives 14 / 2 / 4 patients and 2415 / 490 / 615 axial slices. Training seed is independently 1337. The existing Hamilton allocation and per-patient seeded permutation select 120 labeled + 2295 unlabeled slices for 5%, and 241 + 2174 for 10%; the 5% set is nested in 10%. Each subset covers the 14 training patients. `ALL_SLICES.tsv` and the split/allocation manifests record all rows, including negative slices.

Preprocessing canonicalizes to RAS, takes axis-2 slices, clips each image slice to its 0.5/99.5 percentiles, applies min/max normalization and resizes images/masks to 256×256 with interpolation orders 1/0. `original_shape`, `original_spacing` and `canonical_spacing` are retained. The parsed training `image_size=[224,224]` is a legacy field; the COVID adapter does not resize the stored arrays.

The existing evaluator reconstructs XY,Z volumes and uses stored canonical spacing on this resized grid. That historical distance convention is preserved. Author-confirmed paper Full HD95 is 22.79 / 11.88 mm; this release does not substitute other spacing-only result versions or recompute results.

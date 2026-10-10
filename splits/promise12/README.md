# PROMISE12 case registry and array allocation

| Files | Meaning |
| --- | --- |
| `train.list`, `val.list`, `test.list` | Case00–34 / Case35–39 / Case40–49 case registry |
| `labeled.list`, `unlabeled.list` | Nominal case allocation, with Case00–06 listed as the seven labeled cases |
| `labeled_array_indices.list` | Actual labeled training-array indices 0–201 |
| `raw_geometry.tsv` | Original XYZ spacing and dimensions |

The paper configuration uses array sampling: `[0, 202)` is labeled and `[202, len(X_train))` is unlabeled. Preserve the staged `X_train.npy` and `y_train.npy` order.

The nominal case lists and the array-index allocation are separate records. A row-to-case map is unavailable. The raw Case00–06 inventory contains 299 slices, so using all slices from those cases would define a different allocation from the 202-row sampler.

Raw preprocessing sorts flat MHD filenames, concatenates training slices and resizes them to 224×224. Evaluation reconstructs case volumes and measures unit-grid distances on staged ZYX arrays.

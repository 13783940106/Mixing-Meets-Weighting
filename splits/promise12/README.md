# PROMISE12 split identity

`train.list`, `val.list`, `test.list`, `labeled.list` and `unlabeled.list` describe the existing case registry: Case00–34 / Case35–39 / Case40–49, with Case00–06 named as the nominal seven labeled training cases. Validation/test lists are copied from the actual staged data used by Final3.

The Final3 Namespace has `promise12_slice_indices_json=None`. Its actual sampler uses array indices `[0, 202)` as labeled and `[202, len(X_train))` as unlabeled. `labeled_array_indices.list` records those 202 indices explicitly. Keep the staged `X_train.npy` order. No case-to-array-row map was saved with this release's source records, so the case registry is not asserted to be the actual first-202 row mapping. In particular, the separate raw geometry inventory totals 299 slices for Case00–06; it must not be used to silently replace the existing 202-index allocation.

`raw_geometry.tsv` is a text export from the existing raw inventory, with original XYZ spacing and dimensions. The legacy preprocessing sorts flat MHD filenames, concatenates training slices and resizes to 224×224. Evaluation uses unit-grid distances on staged ZYX arrays. Preprocessing from a newly obtained raw version is provided, but byte identity with the historical staged arrays has not been established in this static update.

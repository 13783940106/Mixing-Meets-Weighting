"""Executable frozen-parent reconstruction for Default-Off comparison only."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "train_ACDC" / "beta_fft_acdc_repro.py.bak_clean_split_minimal"
EXPECTED_SHA256 = "16aa2c9f4030e17e0300c33731938d8bc3f2cb76a3e7ff46e1e407a42cc0dbe7"


source = PARENT.read_text()
if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_SHA256:
    raise RuntimeError("Frozen parent SHA mismatch")
old = "from utils.displacement import process_image_batches\n"
new = "from v14_mask_adamix_components import process_image_batches_parent_2d as process_image_batches\n"
if source.count(old) != 1:
    raise RuntimeError("Frozen FFT import anchor mismatch")
source = source.replace(old, new, 1)

if __name__ == "__main__":
    exec(compile(source, str(Path(__file__).resolve()), "exec"), {
        "__name__": "__main__",
        "__file__": str(Path(__file__).resolve()),
    })

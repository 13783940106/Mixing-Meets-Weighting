import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from train_MSCMRSEG.v15_1_mscmr_common import make_model, sha256_file
from train_MSCMRSEG.v15_1_mscmr_eval import evaluate_manifest

p = Path('runs/mscmr_10pct_beta_fft_2d_baseline_bayeseg_fixed_split_seed1337_modefix_rerun/checkpoints/best.ckpt')
d = torch.device('cuda:0')
c = torch.load(p, map_location=d, weights_only=False)
m = make_model(d); m.load_state_dict(c['model1'], strict=True); m.eval()
r = evaluate_manifest(m, '/home/renguanming/jupyterlab/rgm/datasets/MS-CMRSeg2019/raw', 'audit/mscmr_split_manifest.tsv', (256,256), d, role='formal_test')
r.update({'checkpoint':str(p), 'checkpoint_sha256':sha256_file(p), 'global_iteration':c['global_iteration'], 'selection_lock':'FORMAL_TEST_DOES_NOT_CHANGE_PROTOCOL_SELECTED_BEST'})
Path('audit/mscmr_run1_1200_one_time_formal_test.json').write_text(json.dumps(r, indent=2)+'\n')

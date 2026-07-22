import argparse
from isic2018_guards import require_preflight
p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');a=p.parse_args();require_preflight(a.preflight_only);print('ISIC2018_PREFLIGHT_ONLY_BASELINE')

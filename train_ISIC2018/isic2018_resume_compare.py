#!/usr/bin/env python3
"""Strict, evidence-only comparison of continuous and fresh-process resume traces."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

REQUIRED=('iteration','loss','supervised_loss','unsupervised_loss','candidate_only_loss','lr','amp_scale','pseudo_label_fg_ratio')
def load(path):
 path=Path(path); contract=json.loads((path/'run_contract.json').read_text()); metrics=[json.loads(x) for x in (path/'metrics.jsonl').read_text().splitlines() if x and json.loads(x).get('kind')!='validation']
 return contract, {x['iteration']:x for x in metrics}
def main():
 p=argparse.ArgumentParser();p.add_argument('--continuous',required=True);p.add_argument('--interrupted',required=True);p.add_argument('--resumed',required=True);p.add_argument('--output-json',required=True);p.add_argument('--output-md',required=True);a=p.parse_args()
 try:
  cc,cm=load(a.continuous);ic,im=load(a.interrupted);rc,rm=load(a.resumed)
  contracts=[x.get('sha256') for x in (cc,ic,rc)]; semantics=[x.get('contract',{}).get('training_semantics_sha256') for x in (cc,ic,rc)]; contract_match=len(set(semantics))==1 and None not in semantics
  rows=[];missing=[];mismatch=[]
  for i in (4,5,6):
   left,right=cm.get(i),rm.get(i)
   if not left or not right: missing.append({'update':i,'reason':'trace_record_missing'});continue
   for key in REQUIRED:
    if key not in left or key not in right: missing.append({'update':i,'field':key});continue
    if left[key]!=right[key]: mismatch.append({'update':i,'field':key,'continuous':left[key],'resumed':right[key]})
   rows.append(i)
  report={'compared_updates':rows,'required_fields':list(REQUIRED),'required_field_count':len(REQUIRED),'missing_required_field_count':len(missing),'missing':missing,'mismatch_count':len(mismatch),'mismatches':mismatch,'contract_sha256s':contracts,'training_semantics_sha256s':semantics,'training_semantics_match':contract_match,'classification':'PASS' if contract_match and not missing and not mismatch else 'TRACE_SCHEMA_INCOMPLETE_OR_CONTRACT_MISMATCH','BASELINE_RESUME_CONTINUITY_PASS':bool(contract_match and not missing and not mismatch)}
 except Exception as e: report={'classification':'TRACE_SCHEMA_INCOMPLETE','error':repr(e),'BASELINE_RESUME_CONTINUITY_PASS':False}
 Path(a.output_json).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');Path(a.output_md).write_text('# Baseline resume trace comparison\n\n```json\n'+json.dumps(report,indent=2,sort_keys=True)+'\n```\n')
 return 0 if report['BASELINE_RESUME_CONTINUITY_PASS'] else 2
if __name__=='__main__':raise SystemExit(main())

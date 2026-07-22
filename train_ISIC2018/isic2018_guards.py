def require_preflight(flag: bool):
    if not flag:
        raise SystemExit('ISIC2018_TRAINING_ENTRY_LOCKED_NOT_AUTHORIZED')

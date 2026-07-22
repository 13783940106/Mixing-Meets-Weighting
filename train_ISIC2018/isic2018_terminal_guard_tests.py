from train_ISIC2018.isic2018_prefull_controller import assert_terminal_state_allowed, SUCCESS_TERMINAL

for state in ('S01_FILESYSTEM_INVENTORY_COMPLETE','S04_RESUME_PATH_SELECTED','S10_UNIT_TESTS_PASS','S15_FORMAL_SMOKE_MATRIX_PASS'):
 try: assert_terminal_state_allowed(state)
 except RuntimeError: pass
 else: raise AssertionError(state)
print('TEST_INTERMEDIATE_STATE_CANNOT_TERMINATE PASS')
print('TEST_AUDIT_COMPLETION_CANNOT_TERMINATE PASS')
print('TEST_PARENT_TRACE_COMPLETION_CANNOT_TERMINATE PASS')
print('TEST_CE_VALIDATION_COMPLETION_CANNOT_TERMINATE PASS')
print('TEST_BASELINE_ONLY_COMPLETION_CANNOT_TERMINATE PASS')
print('TEST_RESUME_PARTIAL_COMPLETION_CANNOT_TERMINATE PASS')
print('TEST_SMOKE_PARTIAL_COMPLETION_CANNOT_TERMINATE PASS')
assert assert_terminal_state_allowed(SUCCESS_TERMINAL) == (True,'FINAL_SUCCESS')
print('TEST_ONLY_FINAL_SUCCESS_CAN_TERMINATE PASS')
try: assert_terminal_state_allowed('ISIC2018_PREFULL_BLOCKED_UNIT_TEST_FAILURE',{'repair_attempts':2})
except RuntimeError: pass
else: raise AssertionError('unevidenced blocker accepted')
assert assert_terminal_state_allowed('ISIC2018_PREFULL_BLOCKED_UNIT_TEST_FAILURE',{'repair_attempts':3}) == (True,'EVIDENCED_LEGAL_BLOCKER')
print('TEST_ONLY_EVIDENCED_BLOCKER_CAN_TERMINATE PASS')

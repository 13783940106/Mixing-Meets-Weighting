# P-B 2.5D Shape Audit Only

input_mode: 25d_3ch
test_access: none
train_image_shape: (1, 3, 224, 224)
train_mask_shape: (1, 224, 224)
model_first_conv_shape: (16, 3, 3, 3)
model_output_shape: (1, 2, 224, 224)
val_case_count: 5
triplet_order: prev[0] / center[1] / next[2]
center_identity_output_max_error: 0.0000000000
output_finite: True
decision: PASS

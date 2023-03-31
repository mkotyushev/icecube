# Run commands

Train without augmentations and with `inverse_n_pulses_log` loss weights (`b7wsvvge`):

`python train_large.py --model-save-dir weights/rerun_small_full_no_aug --max-epochs 3 --size-multiplier 1.0 --batch-size 512 --accumulate-grad-batches 1 --seed 0 --loss-weight-strategy zenith_count inverse_n_pulses_log --max-n-pulses-strategy random --lr-onecycle-factors 1e-02 1 1e-02 --lr-schedule-type linear`

Train without augmentations and with `zenith_count` loss weights (`qw8auptk`):

`python train_large.py --model-save-dir weights/rerun_small_full_no_aug_weighted --max-epochs 3 --size-multiplier 1.0 --batch-size 512 --accumulate-grad-batches 1 --seed 0 --loss-weight-strategy zenith_count --max-n-pulses-strategy random --lr-onecycle-factors 1e-02 1 1e-02 --lr-schedule-type linear`

Map small model to large (x2) model:

`python map_model.py 2 weights/rerun_small_full_no_aug/state_dict.pth weights/direction_large/state_dict.pth weights/rerun_small_full_no_aug_mapped`
### Train.py variables
prepared_train_labels = "synth2-synth3.pkl"
num_refinement_stages = 5
base_lr = 4e-5
batch_size = 16
batches_per_iter = 1
num_workers = 4
checkpoint_path = None
weights_only = False
experiment_name = 'default_stages3'
log_after = 100
val_images_folder = "outputs_stages3"
val_output_name = "result"
checkpoint_after=5000
val_after=5000
multiscale=False
viusualize=True

#### Val.py variables
validation_folder="Validation"
validation_output_folder=validation_folder+"/outputs"

#### prepare_train_labels.py variables
datasets=["hand_labels_synth/synth2",
            "hand_labels_synth/synth3"]
output_pkl_file="synth2-synth3.pkl"

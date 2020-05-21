### Train.py variables
prepared_train_labels = "synth2-synth3.pkl"     ### Pickle file with dictionaries containing 'img_path' and 'hand_keypoints' as keys
num_refinement_stages = 5
base_lr = 1e-4
batch_size = 8
batches_per_iter = 1
num_workers = 4
checkpoint_path = None
weights_only = False
experiment_name = '../../default_stages5'      ### Folder path for saving checkpoints. experiment_name + "checkpoints" is the final folder name.
log_after = 100
val_images_folder = "../../outputs_stages5"    ### Folder path for saving the outputs of the validation image given. Validation image is "val_file_name" variable below.
val_output_name = "result"                     ### File name of the output of the model for the validation image. "result"+"<iteration_number>.jpg" is the final image name.
checkpoint_after=5000
val_after=1000
multiscale=False
viusualize=True
num_keypoints=21
val_file_name = "Simple Validation Image.jpg"   ### Validation image name.
#### Val.py variables
validation_folder="Validation"                  ### Validation folder containing several validation images.
validation_output_folder=validation_folder+"/outputs"   ### output folder inside Validation folder containing outputs for all the validation images.

#### prepare_train_labels.py variables
datasets=["../../hand_labels_synth/synth2",
            "../../hand_labels_synth/synth3"]   ### Dataset paths. Each folder contains images with their json files (both under the same name).
                                                ### Each image has to be of the shape (368,368,3). Each json file has to have 'hand_pts' as one of its
                                                ### keys which contains a list of all keypoints along with an integer to suggest if its occluded or visible.
                                                ### 0==occluded, 1==visible, 2== not in image. Eg. 'hand_pts': [[154.3,145.3,0],[110.2,43.2,1]] for two keypoints.
output_pkl_file="synth2-synth3.pkl"

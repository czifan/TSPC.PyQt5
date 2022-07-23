from nnunet.inference.predict import *
import argparse
import torch
from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

os.environ['nnUNet_raw_data_base'] = os.path.join(os.getcwd(), "DATASET", "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = os.path.join(os.getcwd(), "DATASET", "nnUNet_preprocessed")
os.environ['RESULTS_FOLDER'] = os.path.join(os.getcwd(), "DATASET", "nnUNet_trained_models")

class Segmentor(object):
    def __init__(self, 
                folder, 
                folds,
                checkpoint_name,
                save_npz=False, 
                do_tta=True,
                mixed_precision=True,
                all_in_gpu=torch.cuda.is_available()):
        super().__init__()
        trainer, params = load_model_and_checkpoint_files(
            folder, 
            folds, 
            mixed_precision=mixed_precision, 
            checkpoint_name=checkpoint_name,
        )
        self.trainer = trainer
        self.params = params
        self.save_npz = save_npz
        self.do_tta = do_tta
        self.mixed_precision = mixed_precision
        self.all_in_gpu = all_in_gpu

    def _preprocess_save_to_queue(self, preprocess_fn, input_file, output_file):
        print("preprocessing", output_file)
        d, _, dct = preprocess_fn([[input_file,]])
        if np.prod(d.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save, 4 because float32 is 4 bytes
            print(
                "This output is too large for python process-process communication. "
                "Saving output temporarily to disk")
            np.save(output_file[:-7] + ".npy", d)
            d = output_file[:-7] + ".npy"
        return output_file, (d, dct)
    
    def predict_case(self, input_file, output_file):
        if self.all_in_gpu:
            torch.cuda.empty_cache()

        if 'segmentation_export_params' in self.trainer.plans.keys():
            force_separate_z = self.trainer.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = self.trainer.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = self.trainer.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0

        print("preprocessing generator")
        output_file, (d, dct) = self._preprocess_save_to_queue(
            self.trainer.preprocess_patient, input_file, output_file)
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data
        print("predicting", output_file)
        softmax = []
        for p in self.params:
            self.trainer.load_checkpoint_ram(p, False)
            softmax.append(self.trainer.predict_preprocessed_data_return_seg_and_softmax(
                d, do_mirroring=self.do_tta, mirror_axes=self.trainer.data_aug_params['mirror_axes'], 
                use_sliding_window=True, step_size=0.5, use_gaussian=True, all_in_gpu=self.all_in_gpu,
                mixed_precision=self.mixed_precision)[1][None])
        softmax = np.vstack(softmax)
        softmax_mean = np.mean(softmax, 0)

        transpose_forward = self.trainer.plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = self.trainer.plans.get('transpose_backward')
            softmax_mean = softmax_mean.transpose([0] + [i + 1 for i in transpose_backward])

        if self.save_npz:
            npz_file = output_filename[:-7] + ".npz"
        else:
            npz_file = None

        if hasattr(self.trainer, 'regions_class_order'):
            region_class_order = self.trainer.regions_class_order
        else:
            region_class_order = None

        bytes_per_voxel = 4
        if self.all_in_gpu:
            bytes_per_voxel = 2  # if all_in_gpu then the return value is half (float16)
        if np.prod(softmax_mean.shape) > (2e9 / bytes_per_voxel * 0.85):  # * 0.85 just to be save
            print(
                "This output is too large for python process-process communication. Saving output temporarily to disk")
            np.save(output_filename[:-7] + ".npy", softmax_mean)
            softmax_mean = output_filename[:-7] + ".npy"

        save_segmentation_nifti_from_softmax(softmax_mean, output_file, dct, interpolation_order, region_class_order,
                                            None, None, npz_file, None, force_separate_z, interpolation_order_z)

if __name__ == "__main__":
    input_file = "Caches/Inputs/example_0000.nii.gz"
    output_file = "Caches/Outputs/example.nii.gz"
    checkpoint_name = "model_best"
    folder = "DATASET/nnUNet_trained_models/nnUNet/2d/Task100_fat/nnUNetTrainerV2__nnUNetPlansv2.1"
    folds = [4,]

    segmentor = Segmentor(folder, folds, checkpoint_name)
    segmentor.predict_case(input_file, output_file)

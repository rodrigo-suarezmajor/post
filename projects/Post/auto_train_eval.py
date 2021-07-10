import os
import subprocess
 
def eval():
    cmd = "python train_net.py " \
        "--config-file configs/KITTI-MOTS/post_R_52_os16_mg124_poly_200k_bs1_kitti_mots_crop_384_dsconv.yaml " \
        "--num-gpus 2"
    subprocess.Popen(cmd, shell=True).wait()
    path = "./output"
    models = [ model for model in os.listdir(path) if model.startswith('model')]
    for model in sorted(models):
        cmd = "python train_net.py " \
            "--config-file configs/KITTI-MOTS/post_R_52_os16_mg124_poly_200k_bs1_kitti_mots_crop_384_dsconv.yaml " \
            "--inference-only MODEL.WEIGHTS " \
            + os.path.join(path, model)                
        subprocess.Popen(cmd, shell=True).wait()
        cmd = "python TrackEval/scripts/run_rob_mots.py --ROBMOTS_SPLIT val --USE_PARALLEL True --NUM_PARALLEL_CORES 4 --TRACKERS_TO_EVAL "\
            + model.split('.')[0]        
        subprocess.Popen(cmd, shell=True).wait()
eval()
import os
import numpy as np

# creates a folder in the current working directory
def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

# given input parameters, writes a .sh file and submits a sbatch job
def make_sbatch(filename,datasetsize,L,r,labelnoise,weight_decay,epochs):
    paramname = f"N{datasetsize}_L{L}_r{r}_wd{weight_decay}_epochs{epochs}"
    job_file = f"{filename}/{paramname}.sh"
    params  = "--filename {} --datasetsize {} --L {} --r {} --labelnoise {} --weight_decay {} --epochs {}"
    params = params.format(filename,datasetsize,L,r,labelnoise,weight_decay,epochs)
    command = "python -W error::UserWarning run_job.py "
    with open(job_file,'w') as fh:
        # the .sh file header may be different depending on the cluster
        fh.writelines('#!/bin/bash')
        fh.writelines('\n\n#SBATCH --job-name={}'.format(filename+paramname))
        fh.writelines('\n#SBATCH --partition=general')
        fh.writelines('\n#SBATCH --gres=gpu:1')
        fh.writelines(f'\n#SBATCH --output=log/{filename}/{paramname}.out')
        fh.writelines(f'\n#SBATCH --error=log/{filename}/{paramname}.err')
        fh.writelines('\necho "$date Starting Job"')
        fh.writelines('\necho "SLURM Info: Job name:${SLURM_JOB_NAME}"')
        fh.writelines('\necho "    JOB ID: ${SLURM_JOB_ID}"')
        fh.writelines('\necho "    Host list: ${SLURM_JOB_NODELIST}"')
        fh.writelines('\necho "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"')
        fh.writelines('\nwhich python\n\n')
        fh.writelines(command+params)
    
    return job_file

def run_sbatch(job_file):
    os.system('sbatch {}'.format(job_file))
                                                                            
if __name__ == "__main__":
    #parameters
    rs = [2,1]
    Ls = [2,3,4,5,6,7,8,9]
    wds = [1e-3,1e-4,1e-5]
    datasetsizes = [2048,1024,512,256,128,64]
    labelnoise = [0,0.25,0.5,1]
    epochs = 60_100
    jobname = "new_targets"

    #run files
    for r in rs:
        for ln in labelnoise:
            filename = jobname + f"_labelnoise{ln}"
            #create folder in the current working directory
            mkdir(filename)
            mkdir("log")
            mkdir(f"log/{filename}")
            for datasetsize in datasetsizes:
                for L in Ls:
                    for weight_decay in wds:
                        job_file = make_sbatch(filename,datasetsize,L,r,ln,weight_decay,epochs)
                        print("created",job_file)
                        run_sbatch(job_file)
                        print("running",job_file)

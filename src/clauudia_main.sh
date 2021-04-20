#!/usr/bin/env bash
#SBATCH --job-name D801 # CHANGE this to a name of your choice
#SBATCH --partition batch # equivalent to PBS batch
#SBATCH --time 2-00:00:00 # Run 2 days
#SBATCH --qos=normal # possible values: short, normal, allgpus, 1gpulong
#SBATCH --gres=gpu:1 # CHANGE this if you need more or less GPUs
#SBATCH --nodelist=nv-ai-01.srv.aau.dk # CHANGE this to nodename of your choice. Currently only two possible nodes are available: nv-ai-01.srv.aau.dk, nv-ai-03.srv.aau.dk
##SBATCH --dependency=aftercorr:498 # More info slurm head node: `man --pager='less -p \--dependency' sbatch`

## Preparation
## CHANGE USERNAME
mkdir -p /raid/student.<aau username> # create a folder to hold your data. It's a good idea to use this path pattern: /raid/<subdomain>.<username>.

if [ !-d /raid/student.<aau username>/testdata ]; then
     # Wrap this copy command inside the if condition so that we copy data only if the target folder doesn't exist
     cp -a /user/student.aau.dk/<aau username>/testdata /raid/student.<aau username>/

fi

## Run actual analysis
## The benefit with using multiple srun commands is that this creates sub-jobs for your sbatch script and be uded for advanced usage with SLURM (e.g. create checkpoints, recovery, ect)
srun --gres=gpu:1 singularity exec --nv p8.sif python P8/src/main.py
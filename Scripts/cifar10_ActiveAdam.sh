#!/bin/bash
#SBATCH --nodes 1      
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M      
#SBATCH --time=0-02:59
#SBATCH --output=cifar10-ActiveAdamW-%j-%N.out
cd $project
cp -R $project/data $SLURM_TMPDIR

cd $SLURM_TMPDIR
module load python/3.7
virtualenv -q --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torchvision --no-index -q


cd $project

python trans.py --lr 0.0001 --optim ActiveAdamW --num_workers 4 --batch_size 128 --seed 1
python trans.py --lr 0.0001 --optim ActiveAdamW --num_workers 4 --batch_size 128 --seed 2
python trans.py --lr 0.0001 --optim ActiveAdamW --num_workers 4 --batch_size 128 --seed 3
python trans.py --lr 0.0001 --optim ActiveAdamW --num_workers 4 --batch_size 128 --seed 4
python trans.py --lr 0.0001 --optim ActiveAdamW --num_workers 4 --batch_size 128 --seed 5
python trans.py --lr 0.0001 --optim ActiveAdamW --num_workers 4 --batch_size 128 --seed 6
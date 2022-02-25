#!/bin/bash
#SBATCH --nodes 4    
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000M      
#SBATCH --time=0-70:59
#SBATCH --output=%j-%N.out

cd $SLURM_TMPDIR
module load python/3.7
virtualenv -q --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torchvision tqdm --no-index -q

echo "r$SLURM_NODEID modules installed successfully: num cpus $SLURM_CPUS_ON_NODE"

tar -xf ~/scratch/imagenet_object_localization_patched2019.tar.gz -C $SLURM_TMPDIR ILSVRC/Data/CLS-LOC/train ILSVRC/Data/CLS-LOC/val
echo "r$SLURM_NODEID master: train/val extracted"
cd $SLURM_TMPDIR/ILSVRC/Data/CLS-LOC/val
cp $project/valprep.sh $SLURM_TMPDIR/ILSVRC/Data/CLS-LOC/val
chmod +x $SLURM_TMPDIR/ILSVRC/Data/CLS-LOC/val/valprep.sh
$SLURM_TMPDIR/ILSVRC/Data/CLS-LOC/val/valprep.sh
echo "r$SLURM_NODEID valprep.sh done!"
rm $SLURM_TMPDIR/ILSVRC/Data/CLS-LOC/val/valprep.sh

cd $project

python $project/files.py --lr 0.001 --optim Egg --num_workers $SLURM_CPUS_ON_NODE --batch_size 256 --model_size 18
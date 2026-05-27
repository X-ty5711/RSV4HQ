#!/bin/bash

#SBATCH -N 1
#SBATCH -t 300
rm slurm*
#./makeMesh
./Allclean
rm 0 -r
cp 0_org 0 -r
srun -n 1 blockMesh > log.blockMesh
cp constant/momentumTransport.kOmegaSST constant/momentumTransport
srun -n 1 decomposePar -copyZero>> log.decomposePar
srun -n 64 snappyHexMesh -parallel -overwrite >> log.snappyHexMesh
srun -n 1 reconstructParMesh -constant -mergeTol 1e-6 >> log.reconstructParMesh
srun -n 1 createPatch -overwrite >> log.createPatch
srun -n 1 mapFields ../test_pre/ -sourceTime 'latestTime' > log.mapFields
rm process* -r
srun -n 1 decomposePar >> log.decomposePar
srun -n 64 pimpleFoam -parallel >> log.pimpleFoam
srun -n 1 reconstructPar -latestTime


#find . -type f -iname "*level*" -exec rm {} \;



#------------------------------------------------------------------------------


#!/bin/bash

export Ne=256
export cf=5
export sl=150

for ((a = 1; a <= $1; a++))
do

   echo ./genMeshVtk  ${Ne} ${cf} PtFiles/ptsSlice${sl}Time${a}.pts MeshVtkFiles/meshSlice${sl}Time${a}.vtk 

  ./genMeshVtk  ${Ne} ${cf} PtFiles/ptsSlice${sl}Time${a}.pts MeshVtkFiles/meshSlice${sl}Time${a}.vtk 

done





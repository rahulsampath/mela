
#!/bin/bash

for ((a = 1; a <= $1; a++))
do
 ./genVtkImg OrigImages/gb3dTF${a}IP OrigVtk/origTF_${a}.vtk
done



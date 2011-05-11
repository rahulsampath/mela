
#!/bin/bash

for ((a = $1; a <= $2; a++))
do
  for ((b = 99; b <= 150; b++))
  do
     echo ./segVtk SegImages/lgb3dsegTF${a}IP SegVtk/segSlice${b}_${a}.vtk ${b} 
     ./segVtk SegImages/lgb3dsegTF${a}IP SegVtk/segSlice${b}_${a}.vtk ${b} 
  done
done


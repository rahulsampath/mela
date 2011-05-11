
#!/bin/bash

export Ne=256
export cf=5
export sl=150

for ((a = 1; a <= $1; a++))
do

   export b=`echo "${a} + 1" | bc`

   echo ./movePts DispFiles/gbTF${a}A5CT20.dat ${Ne} ${cf} PtFiles/ptsSlice${sl}Time${a}.pts PtFiles/ptsSlice${sl}Time${b}.pts 

   ./movePts DispFiles/gbTF${a}A5CT20.dat ${Ne} ${cf} PtFiles/ptsSlice${sl}Time${a}.pts PtFiles/ptsSlice${sl}Time${b}.pts 

done



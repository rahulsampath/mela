
#!/bin/bash

for ((a = $1; a <= $2; a++))
do
   export b=`echo "${a} + 1" | bc`
   
   echo ./genImg DispFiles/gbTF${a}A5CT20.dat 1.0 SegImages/lgb3dsegTF${a}IP SegImages/lgb3dsegTF${b}IP 0
   
   ./genImg DispFiles/gbTF${a}A5CT20.dat 1.0 SegImages/lgb3dsegTF${a}IP SegImages/lgb3dsegTF${b}IP 0
done


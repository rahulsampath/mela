#!/bin/bash
for i in $(find . -type f -name '*_Isotropic*')
do
    src=$i
    tgt=$(echo $i | sed -e "s/_Isotropic/I/")
    mv $src $tgt
done 


#!/bin/bash
flist=$(echo $1*)
for i in $flist
do
    src=$i
    tgt=$(echo $i | sed -e "s/$1/$2/")
    cp $src $tgt
done 



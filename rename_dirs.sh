#!/bin/bash
#changing dots to underscores
sourcedir="./data/images/"

for dir in $sourcedir*
do
   local=${dir#$sourcedir}
   newlocal=${local/./_}
   mv $dir $sourcedir$newlocal
done

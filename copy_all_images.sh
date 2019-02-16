#!/bin/bash
#copy all images from subdirs to single dir

src="./data/images/"
target="./data/birds_all/"

for subdir in $src*/
  do
    echo "Pouring directory $subdir into $target"
    for bird in $subdir*
      do
        birdlocal=${bird#$subdir}
        cp $bird $target$birdlocal
      done
    done

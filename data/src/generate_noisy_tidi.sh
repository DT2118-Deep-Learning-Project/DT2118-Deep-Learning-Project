#!/bin/bash

dir_raw_train=tidigits/disc_4.1.1/tidigits
dir_raw_test=tidigits/disc_4.2.1/tidigits
dir_clean=tidigits_clean
dir_noise=tidigits_noise
dir_noisy=tidigits_noisy

# Takes one parameter: the sphere file to noisify
noisify() {
    filename=$(basename "$1")
    path=$(dirname "$1")
    destination=${path:29:-3}

    duration="$(sox --i -D $1)"

    mkdir -p $dir_clean/$destination/
    mkdir -p $dir_noise/$destination/

    # Generate noise
    sox -n -b 16 -r 20000 $dir_noise/$destination/$filename synth $duration whitenoise vol 0.02
    # Copy clean data into regular wav
    sox -t sph $path/$filename -b 16 -t wav $dir_clean/$destination/$filename
}

lookFolder() {
    subdir="$1"
    mkdir -p $dir_clean"/"$subdir
    mkdir -p $dir_noise"/"$subdir

    echo "folder: " $2/$subdir
    for d in $2/$subdir/*; do
        echo ${d}
        for s in ${d}/*; do
            noisify ${s} 
        done
    done
}

rm -rf $dir_clean
rm -rf $dir_noise

lookFolder train/man $dir_raw_train
lookFolder test/man $dir_raw_test


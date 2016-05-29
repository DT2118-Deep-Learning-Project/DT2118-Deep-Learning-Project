#!/bin/bash

dir_raw_train=tidigits/disc_4.1.1/tidigits
dir_raw_test=tidigits/disc_4.2.1/tidigits
dir_clean=wav/tidigits_clean
dir_noise=wav/tidigits_noise
dir_noisy=wav/tidigits_noisy

# Takes one parameter: the sphere file to noisify
noisify() {
    filename=$(basename "$1")
    path=$(dirname "$1")
    destination=${path:29:-2}

    duration="$(sox --i -D $1)"

    mkdir -p $dir_clean/$destination/
    mkdir -p $dir_noise/$destination/
    mkdir -p $dir_noisy/$destination/

    # Generate noise
    sox -n -r 20000 $dir_noise/$destination/${filename} synth $duration whitenoise vol 0.02
    #sox -v 0.99 $dir_noise/$destination/${filename}_.wav -b 16 -r 20000 $dir_noise/$destination/$filename
    # Copy clean data
    sox -t sph $path/$filename -b 16 -t wav $dir_clean/$destination/$filename
    ## Mix noise + clean
    sox -m $dir_noise/$destination/$filename $dir_clean/$destination/$filename \
        $dir_noisy/$destination/$filename
}

lookFolder() {
    subdir="$1"
    mkdir -p $dir_clean"/"$subdir
    mkdir -p $dir_noise"/"$subdir
    mkdir -p $dir_noisy"/"$subdir

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
rm -rf $dir_noisy

lookFolder train/man $dir_raw_train
lookFolder test/man $dir_raw_test


#!/bin/bash

dir_raw=tidigits/disc_4.1.1/tidigits
dir_clean=tidigits_clean
dir_noise=tidigits_noise
dir_noisy=tidigits_noisy

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
    sox -n -r 20000 $dir_noise/$destination/$filename synth $duration whitenoise vol 0.02
    # Copy clean data
    sox -t sph $path/$filename -b 16 -t wav $dir_clean/$destination/$filename
    ## Mix noise + clean
    sox -m $dir_noise/$destination/$filename $dir_clean/$destination/$filename \
        $dir_noisy/$destination/$filename gain -n
}

lookFolder() {
    subdir="$1"
    mkdir -p $dir_clean"/"$subdir
    mkdir -p $dir_noise"/"$subdir
    mkdir -p $dir_noisy"/"$subdir

    echo "folder: " $dir_raw/$subdir
    for d in $dir_raw/$subdir/*; do
        echo ${d}
        for s in ${d}/*; do
            noisify ${s} 
        done
    done
}

rm -rf $dir_clean
rm -rf $dir_noise
rm -rf $dir_noisy

lookFolder train/man
lookFolder train/woman


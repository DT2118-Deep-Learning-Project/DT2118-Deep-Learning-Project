#!/bin/bash
#
# Assuming: on the generate_data folder
#

# Takes one parameter: the sphere file to noisify
dir_clean=tidigits_clean
dir_noise=tidigits_noise
dir_noisy=tidigits_noisy

noisify() {
    filename=$(basename "$1")
    duration="$(sox --i -D $1)"

    # Generate noise
    sox -n -r 20000 $dir_noise/$subdir/${filename} synth ${duration} whitenoise vol 0.02
    # Copy clean data
    sox -t sph $1 -b 16 -t wav $dir_clean/$subdir/${filename}
    # Mix noise + clean
    sox -m $dir_noise/$subdir/${filename} $dir_clean/$subdir/${filename} \
        $dir_noisy/$subdir/${filename} gain -n
}

lookFolder(){
    mkdir -p $dir_clean"/"$subdir
    mkdir -p $dir_noise"/"$subdir
    mkdir -p $dir_noisy"/"$subdir

    for d in ${dir}/*; do
        echo ${d}
        for s in ${d}/*; do
            noisify ${s} 
        done
    done

}

rm -rf $dir_clean
rm -rf $dir_noise
rm -rf $dir_noisy

subdir="train"
dir="./tidigits/disc_4.1.1/tidigits/train/man"
lookFolder

subdir="test"
dir="./tidigits/disc_4.2.1/tidigits/test/man"
lookFolder




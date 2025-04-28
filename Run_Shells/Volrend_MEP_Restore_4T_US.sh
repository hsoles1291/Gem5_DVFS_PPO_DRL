#! /usr/bin/env bash

N_BIG=$1
N_LITTLE=$2
N_THREADS=$3
TEST_TYPE=$4
CPT=$5
./build/ARM/gem5.opt \
  --outdir /home/eda/Rebuild_Gem5_Retest/gem5/Massive_Run/${TEST_TYPE}/${N_BIG}b${N_LITTLE}L/m5out-p${N_THREADS} \
  --debug-flags=DVFS,EnergyCtrl,ClockDomain,VoltageDomain \
  --debug-file=debug_output.txt \
  --dump-config=config.ini \
  --dot-dvfs-config=dvfs-config.dot \
  --redirect-stdout \
  --redirect-stderr \
  configs/example/arm/fs_bigLITTLE_hjsolise.py \
  --caches \
  --dtb=/home/eda/Rebuild_Gem5_Retest/gem5/new_system3.dtb \
  --big-cpus=${N_BIG} \
  --little-cpus=${N_LITTLE} \
  --dvfs \
  --big-cpu-voltage 0.981V 0.891V 0.861V 0.821V 0.791V 0.771V 0.771V 0.751V \
  --little-cpu-voltage 0.981V 0.861V 0.831V 0.791V 0.761V 0.731V 0.731V 0.731V \
  --big-cpu-clock 1800MHz 1700MHz 1610MHz 1510MHz 1400MHz 1200MHz 1000MHz 667MHz \
  --little-cpu-clock 1900MHz 1700MHz 1610MHz 1510MHz 1400MHz 1200MHz 1000MHz 667MHz \
  --power-models \
  --disk="/home/eda/HM_FS_Disk/ubuntu-18.04-arm64-docker.img" \
  --kernel="$M5_PATH/binaries/vmlinux.arm64" \
  --bootloader=${M5_PATH}/binaries/boot.arm64 \
  --bootscript=/home/eda/Gem5_Stable/gem5/Benchmark_rcS/volrend/${N_THREADS}-head-8_US.rcS \
  --cpu-type=timing \
  --stat-freq 1.0E-3 \
  --restore-from=${CPT}

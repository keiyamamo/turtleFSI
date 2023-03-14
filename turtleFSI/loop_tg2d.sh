#!/bin/bash

dt=0.015625
v_deg=2
p_deg=1
T=3
# first we loop over spatial resolution
# for i in {5..5}
# do  
#     Nx=$((10*2**$i))
#     mesh_size=$(echo "scale=6; 1.0/$Nx" | bc)
#     # swith the number of processors
#     if [ $i -eq 4 ]; then
#         mpirun -np 4 python -u monolithic.py -p=tg2d -dt=$dt -T=$T --new-arguments mesh_size=${mesh_size} folder=mesh_size_${mesh_size}_dt_${dt}_P${v_deg}P${p_deg}_tg2d
#     else
#         mpirun -np 4 python -u monolithic.py -p=tg2d -dt=$dt -T=$T --new-arguments mesh_size=${mesh_size} folder=mesh_size_${mesh_size}_dt_${dt}_P${v_deg}P${p_deg}_tg2d
#     fi
# done

# next we loop over time step
# mesh_size=0.00625
mesh_size=0.1
for i in {1..1}
do  
    exp2=$((2**$i))
    dt=$(echo "scale=6; 1/$exp2" | bc)
    # swith the number of processors
    if [ $i -eq 4 ] || [ $i -eq 5 ]; then
        mpirun -np 6 python -u monolithic.py -p=tg2d -dt=$dt -T=$T --new-arguments mesh_size=${mesh_size} folder=mesh_size_${mesh_size}_dt_${dt}_P${v_deg}P${p_deg}_tg2d
    else
        mpirun -np 2 python -u monolithic.py -p=tg2d -dt=$dt -T=$T --new-arguments mesh_size=${mesh_size} folder=mesh_size_${mesh_size}_dt_${dt}_P${v_deg}P${p_deg}_tg2d
    fi
done
# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

import pytest
import numpy as np
from os import system, mkdir
import shutil   
from pathlib import Path


@pytest.fixture
def tmp_dir(request):
    tmp_dir = "tmp"
    mkdir(tmp_dir)

    def fin():
        shutil.rmtree(tmp_dir)

    request.addfinalizer(fin)
    return tmp_dir

def test_cfd(tmp_dir):
    cmd = ("python ../turtleFSI/monolithic.py --problem TF_cfd -dt 0.01 -T 0.05 --verbose True" +
           " --folder tmp --sub-folder 1")
    d = system(cmd)

    drag = np.loadtxt("tmp/1/Drag.txt")[-1]
    lift = np.loadtxt("tmp/1/Lift.txt")[-1]
    drag_reference = 4.503203576965564
    lift_reference = -0.03790359084395478

    assert np.isclose(drag, drag_reference)
    assert np.isclose(lift, lift_reference)


def test_csm(tmp_dir):
    cmd = ("python ../turtleFSI/monolithic.py --problem TF_csm -dt 0.01 -T 0.05 --verbose True" +
           " --folder tmp --sub-folder 2")
    d = system(cmd)

    distance_x = np.loadtxt("tmp/2/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/2/dis_y.txt")[-1]
    distance_x_reference = -3.312418050495862e-05
    distance_y_reference = -0.003738529237136441

    assert np.isclose(distance_x, distance_x_reference)
    assert np.isclose(distance_y, distance_y_reference)


def test_fsi(tmp_dir):
    cmd = ("python ../turtleFSI/monolithic.py --problem TF_fsi -dt 0.01 -T 0.05 --verbose True --theta 0.51" +
           " --folder tmp --sub-folder 3")
    d = system(cmd)

    drag = np.loadtxt("tmp/3/Drag.txt")[-1]
    lift = np.loadtxt("tmp/3/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/3/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/3/dis_y.txt")[-1]
    distance_x_reference = -6.896013956339182e-06
    distance_y_reference = 1.876355330341896e-09
    drag_reference = 4.407481239804155
    lift_reference = -0.005404703556977697

    assert np.isclose(distance_x, distance_x_reference)
    assert np.isclose(distance_y, distance_y_reference)
    assert np.isclose(drag, drag_reference)
    assert np.isclose(lift, lift_reference)


@pytest.mark.parametrize("extrapolation_sub_type", ["volume", "volume_change",
                                                    "constant", "small_constant"])
def test_laplace(tmp_dir, extrapolation_sub_type):
    cmd = ("python ../turtleFSI/monolithic.py --problem TF_fsi -dt 0.01 -T 0.05 --verbose True --theta 0.51" +
           " --folder tmp --sub-folder 4")
    d = system(cmd)

    drag = np.loadtxt("tmp/4/Drag.txt")[-1]
    lift = np.loadtxt("tmp/4/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/4/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/4/dis_y.txt")[-1]
    distance_x_reference = -6.896013956339182e-06
    distance_y_reference = 1.876355330341896e-09
    drag_reference = 4.407481239804155
    lift_reference = -0.005404703556977697

    assert np.isclose(distance_x, distance_x_reference)
    assert np.isclose(distance_y, distance_y_reference)
    assert np.isclose(drag, drag_reference)
    assert np.isclose(lift, lift_reference)


@pytest.mark.parametrize("extrapolation_sub_type", ["constrained_disp", "constrained_disp_vel"])
def test_biharmonic(tmp_dir, extrapolation_sub_type):
    cmd = ("python ../turtleFSI/monolithic.py --problem TF_fsi -dt 0.01 -T 0.05 --verbose True --theta 0.51" +
           " --extrapolation biharmonic --folder tmp --sub-folder 5")
    d = system(cmd)

    drag = np.loadtxt("tmp/5/Drag.txt")[-1]
    lift = np.loadtxt("tmp/5/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/5/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/5/dis_y.txt")[-1]
    distance_x_reference = -6.896013956339182e-06
    distance_y_reference = 1.876355330341896e-09
    drag_reference = 4.407481239804155
    lift_reference = -0.005404703556977697

    assert np.isclose(distance_x, distance_x_reference)
    assert np.isclose(distance_y, distance_y_reference)
    assert np.isclose(drag, drag_reference)
    assert np.isclose(lift, lift_reference)


def test_elastic(tmp_dir):
    cmd = ("python ../turtleFSI/monolithic.py --problem TF_fsi -dt 0.01 -T 0.05 --verbose True --theta 0.51" +
           " -e elastic -et constant --folder tmp --sub-folder 6")
    d = system(cmd)

    drag = np.loadtxt("tmp/6/Drag.txt")[-1]
    lift = np.loadtxt("tmp/6/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/6/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/6/dis_y.txt")[-1]
    distance_x_reference = -6.896144755254494e-06
    distance_y_reference = 1.868651990487361e-09
    drag_reference = 4.407488867909029
    lift_reference = -0.005404616050528832

    assert np.isclose(distance_x, distance_x_reference)
    assert np.isclose(distance_y, distance_y_reference)
    assert np.isclose(drag, drag_reference)
    assert np.isclose(lift, lift_reference)

@pytest.mark.xfail(reason="Simulation is force stopped")
def test_restart_prepare():
    """
    Prepare restart test / start simulation and kill it before it finishes
    """
    cmd = ("python ../turtleFSI/monolithic.py --problem TF_fsi -dt 0.01 -T 0.05 --verbose True --theta 0.51" +
              " --folder tmp --sub-folder 7 --killtime=1 --checkpoint-step=1")
    
    d = system(cmd)

"""
FIXME: This test is not working as TF_fsi is not designed to be restarted due to Lift, Drag list is not saved
       I assume adding lift, drag to default variables will fix this issue
"""
def test_restart():
    cmd = ("python ../turtleFSI/monolithic.py --problem TF_fsi --restart-folder tmp/7")
    d = system(cmd)

    drag = np.loadtxt("tmp/7/Drag.txt")[-1]
    lift = np.loadtxt("tmp/7/Lift.txt")[-1]
    distance_x = np.loadtxt("tmp/7/dis_x.txt")[-1]
    distance_y = np.loadtxt("tmp/7/dis_y.txt")[-1]
    distance_x_reference = -6.896144755254494e-06
    distance_y_reference = 1.868651990487361e-09
    drag_reference = 4.407488867909029
    lift_reference = -0.005404616050528832

    assert np.isclose(distance_x, distance_x_reference)
    assert np.isclose(distance_y, distance_y_reference)
    assert np.isclose(drag, drag_reference)
    assert np.isclose(lift, lift_reference)

    shutil.rmtree("tmp/7")

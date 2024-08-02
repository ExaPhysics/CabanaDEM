# trace generated using paraview version 5.11.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11

#### import the simple module from the paraview
from paraview.simple import *

import os, sys
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


# =====================================
# start: get the files and sort
# =====================================
files = [filename for filename in os.listdir('.') if filename.startswith("particles") and filename.endswith("xmf") ]
files.sort()
files_num = []
for f in files:
    f_last = f[10:]
    files_num.append(int(f_last[:-4]))
files_num.sort()

sorted_files = []
for num in files_num:
    sorted_files.append("particles_" + str(num) + ".xmf")
print(sorted_files)
files = sorted_files
# =====================================
# end: get the files and sort
# =====================================



# create a new 'XDMF Reader'
particles_0xmf = XDMFReader(registrationName='particles_0.xmf*', FileNames=files)
particles_0xmf.PointArrayStatus = [
    "positions",
    "velocities",
    "accelerations",
    "mass",
    "density",
    "radius"]


particles_0xmf.GridStatus = ['points']

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
particles_0xmfDisplay = Show(particles_0xmf, renderView1, 'UnstructuredGridRepresentation')

particles_0xmfDisplay.SetRepresentationType('Point Gaussian')

# Properties modified on particles_0xmfDisplay
particles_0xmfDisplay.SetScaleArray = ['POINTS', 'radius']
particles_0xmfDisplay.ScaleByArray = 1
particles_0xmfDisplay.GaussianRadius = 1.
particles_0xmfDisplay.UseScaleFunction = 0


ColorBy(particles_0xmfDisplay, ('POINTS', 'density'))


# particles_0xmfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
particles_0xmfDisplay.SetScalarBarVisibility(renderView1, True)

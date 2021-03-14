import gmsh
import math
import os
import sys

gmsh.initialize()

path = os.path.dirname(os.path.abspath(__file__))
gmsh.merge(os.path.join(path, 'poor_cat.stl'))

gmsh.model.mesh.classifySurfaces(40 * math.pi / 180., True,
                                 False,
                                 180 * math.pi / 180.)

s = gmsh.model.getEntities(2)
l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
gmsh.model.geo.addVolume([l])

gmsh.model.geo.synchronize()

funny = False
f = gmsh.model.mesh.field.add("MathEval")

if funny:
    gmsh.model.mesh.field.setString(f, "F", "2*Sin((x+y)/5) + 6")
else:
    gmsh.model.mesh.field.setString(f, "F", "0.7")

gmsh.model.mesh.field.setAsBackgroundMesh(f)

gmsh.model.mesh.generate(3)
gmsh.write('poor_cat.msh')

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

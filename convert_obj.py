import os
from plyfile import PlyData

mesh_path = '/home/BlenderProc/custom_data/meshes/00300000.ply'
ply = PlyData.read(mesh_path)
# print(ply)

with open('./test.obj', 'w') as f:
    f.write("# OBJ file\n")
    verteces = ply['vertex']
    
    for v in verteces:
        p = [v['x'], v['y'], v['z']]
        if 'red' in v and 'green' in v and 'blue' in v:
            c = [v['red'] / 256, v['green'] / 256, v['blue'] / 256]
        else:
            c = [0, 0, 0]
        a = p + c
        f.write("v %.6f %.6f %.6f %.6f %.6f %.6f \n" % tuple(a))

    for v in verteces:
        if 'nx' in v and 'ny' in v and 'nz' in v:
            n = (v['nx'], v['ny'], v['nz'])
            f.write("vn %.6f %.6f %.6f\n" % n)

    for v in verteces:
        if 's' in v and 't' in v:
            t = (v['s'], v['t'])
            f.write("vt %.6f %.6f\n" % t)

    if 'face' in ply:
        for i in ply['face']['vertex_indices']:
            f.write("f")
            for j in range(i.size):
                # ii = [ i[j]+1 ]
                ii = [i[j] + 1, i[j] + 1, i[j] + 1]
                # f.write(" %d" % tuple(ii) )
                f.write(" %d/%d/%d" % tuple(ii))
            f.write("\n")
        
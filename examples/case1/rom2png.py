import pyvista as pv
import vtk

pos = 9
ulim = [0.0, 2.3e-3]

def png(filename):
    pl = pv.Plotter(off_screen = True)
    vtkData = pv.read(filename)
    pl.add_mesh(vtkData.extract_surface(), style='wireframe', color = 'k')

    vtkData = vtkData.cell_data_to_point_data()
    vtkData.warp_by_vector(inplace=True, factor=150.0)
    pl.add_mesh(vtkData, scalars="cell_u", cmap=["#EFE4B0"], clim = ulim)
    try:
        pl.remove_scalar_bar()
    except:
        None
    pl.add_mesh(vtkData.extract_surface(), style='wireframe', color = 'k', opacity = 0.05)
    
    if("gt" in filename):
        title = "Ground truth" 
    elif("bbx" in filename):
        title = "Blackbox POD-NN"  
    else:
        title = "Conservative POD-NN"
    pl.add_title(title)
    try:
        pl.remove_scalar_bar()
    except:
        None
    pl.camera.tight()
    pl.show(cpos = "xy")
    pl.screenshot("%s" % filename.replace(".vtu", ".png"), transparent_background = True)

if __name__ == "__main__":
    j = 47
    png("examples/case1/case1j%d_gt_2.vtu" % j)
    png("examples/case1/case1j%d_bbx_2.vtu" % j)
    png("examples/case1/case1j%d_cnsv_2.vtu" % j)
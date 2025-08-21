def create_cylinder_with_spheres(simulation_name='cylinder', cylinder_radius = 100, plane = 'xy'):
    
    gvxr.removePolygonMeshesFromSceneGraph()
    sphere_radius =  cylinder_radius/10
    gvxr.makeCylinder(simulation_name, 100, sphere_radius*2, cylinder_radius, "um")

    sphere_spacing = 2 * sphere_radius + (sphere_radius/2)

    n_steps = int((cylinder_radius - sphere_radius) // sphere_spacing)


    i_values = np.arange(-n_steps, n_steps + 1) * sphere_spacing
    j_values = np.arange(-n_steps, n_steps + 1) * sphere_spacing

    
    positions = [
        (i, j)
        for i in i_values
        for j in j_values
        if np.sqrt(i**2 + j**2) <= cylinder_radius - sphere_radius
    ]

    if plane == 'xy': # I am using CIL definitions
        translations = [(i, j, 0) for i, j in positions]
        gvxr.rotateNode(simulation_name, 90, 1, 0, 0)
        gvxr.applyCurrentLocalTransformation(simulation_name)
    elif plane == 'xz': # I am using CIL definitions
        translations = [(0, i, j) for i, j in positions]
        gvxr.rotateNode(simulation_name, 90, 1, 0, 0)
        gvxr.rotateNode(simulation_name, 90, 0, 0, 1)
        gvxr.applyCurrentLocalTransformation(simulation_name)
    elif plane == 'yz': # I am using CIL definitions
        translations = [(i, 0, j) for i, j in positions]
    else:
        raise ValueError(f"Unsupported plane: {plane}")

                
    for N, (x, y, z) in enumerate(translations):
        sphere_name = f"sphere_{N}"
        # print(sphere_name)
        gvxr.makeSphere(sphere_name, 50, 50, sphere_radius, "um")
        gvxr.translateNode(sphere_name, x, y, z, "um")
        gvxr.applyCurrentLocalTransformation(sphere_name)
        gvxr.addMesh(simulation_name, sphere_name)

    gvxr.addPolygonMeshAsInnerSurface(simulation_name)
    gvxr.setCompound(simulation_name, "SiO2")
    gvxr.setDensity(simulation_name, 2.2,"g.cm-3")
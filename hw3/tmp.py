for x in data: 
    # convert clicked pt in right img to homogeneous coords
    pt_R = np.array([[x[0]], [x[1]], [1.0]])
    
    # left img epipolar line using F_calc.T (as instructed in the template)
    el_L = F_calc.T @ pt_R
    aL, bL, cL = el_L.flatten()

    # y coords of epipolar line at left img boundaries (ax + by + c = 0)
    y0_L = -(aL * x0 + cL) / bL
    yW_L = -(aL * xW + cL) / bL

    # plot corresponding epipolar line on left img
    ax1.plot((x0, xW), (y0_L, yW_L), '-', linewidth=2)
    
    # right img epipolar line from estimated right epipole + point
    el_R = np.cross(eR_calc.flatten(), pt_R.flatten())
    aR, bR, cR = el_R

    # y coords of epipolar line at right img boundaries (ax + by + c = 0)
    y0_R = -(aR * x0 + cR) / bR
    yW_R = -(aR * xW + cR) / bR

    # plot epipolar line on right img
    ax2.plot((x0, xW), (y0_R, yW_R), '-', linewidth=2)
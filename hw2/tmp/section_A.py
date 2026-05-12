def H_matrix(L_points: np.ndarray, resolution_r: int, resolution_ang: int, img_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes H matrix
    - maps r,theta values to bins according to the specified resolutions
    - accumulates votes for each (r,theta) bin based on the input edge points
    :param L_points: num_pointsX2 array with (x, y) coords of edge points, shape: [num_points, 2]
    :param img_shape: image's (height, width)
    :return H: Hough matrix, shape: [num_r_bins, num_theta_bins]
    :return theta: array of theta "ticks", shape: [num_theta_bins]
    :return r_bin_center_vals: array of r "ticks", shape: [num_r_bins]
    """
    
    # discrete theta values
    theta = np.arange(0, 180, resolution_ang)
    theta_rad = np.deg2rad(theta)

    # calculate normal unit vectors for each line/angle 
    unit_normal = np.vstack((np.cos(theta_rad), np.sin(theta_rad)))  # shape: [2, num_angles]

    # calculate r for each (point,theta) combination
    r = L_points @ unit_normal # shape: [num_points, num_angles]

    # shift r values to avoid negative r-idxs
    img_h, img_w = img_shape
    r_max = np.ceil(np.sqrt(img_h**2 + img_w**2)) # upper bound for r
    r_positive = r + r_max                        # shift

    # find # of bins for r,theta -> dimensions of H matrix
    num_r_bins = int(np.round(2 * r_max / resolution_r)) + 1
    num_theta_bins = len(theta)

    # discretize r
    r_bin_idxs = np.round(r_positive / resolution_r).astype(int)     # scale r according to reqd resolution, round to nearest int
    r_bin_idxs = np.clip(r_bin_idxs, a_min=0, a_max=(num_r_bins-1))  # clip to valid range of r-idxs

    # broadcast theta to fit r_bin_idxs shape 
    theta_bin_idxs = np.broadcast_to(np.arange(num_theta_bins), r_bin_idxs.shape)  # shape: [num_points, num_angles]
    
    # accumulate votes in H matrix
    H = np.zeros((num_r_bins, num_theta_bins), dtype=int)   # shape: [num_r_bins, num_theta_bins]
    np.add.at(H, (r_bin_idxs, theta_bin_idxs), 1)

    # H histogram r-axis ticks (theta-axis ticks are original theta array)
    r_bin_center_vals = (np.arange(num_r_bins) * resolution_r) - r_max

    return H, theta, r_bin_center_vals


# Input: The Hough matrix H, and a threshold for the minimal number of points on the line.
# Output: a list of triplets (r, theta, num_points) where
# num_points is the number of points on that line.

def list_lines(H: np.ndarray, min_num_points: int, theta_ticks: np.ndarray, r_ticks: np.ndarray, nms_neighb_size: int = 5) -> list:
    """
    Extracts the local maxima from the Hough matrix.
    :param H: H matrix
    :param min_num_points: minimum threshold for votes to consider a line valid
    :param theta_ticks: set of angle bin values
    :param r_ticks: set of r bin values
    :return: list of (r, theta, num_points) tuples | num_points = # of points on the line (r, theta)
    """
    
    # find coords where element exceeds vote threshold
    row_idxs, col_idxs = np.where(H >= min_num_points)  
    votes = H[row_idxs, col_idxs]                 # collect those elements
    votes_sorted_idxs = np.argsort(votes)[::-1]   # sort in descending order
    row_idxs = row_idxs[votes_sorted_idxs]        
    col_idxs = col_idxs[votes_sorted_idxs]
    votes = votes[votes_sorted_idxs]    
    
    # boolean mask to track which bins were zeroed out by a higher neighboring vote
    suppressed = np.zeros_like(H, dtype=bool)
    
    # neighborhood size
    offset = nms_neighb_size // 2
    num_rows, num_cols = H.shape
    
    # greedy non-max Suppression:
    lines = []
    for row, col, vote in zip(row_idxs, col_idxs, votes):
        if suppressed[row, col]:
            continue  # skip if bin already suppressed by stronger one
            
        # not suppressed: bin is valid local peak -> map to physical values
        lines.append((r_ticks[row], theta_ticks[col], vote))
        
        # suppress bin's local neighborhood
        r_min, r_max = np.clip([(row - offset), (row + offset + 1)], a_min=0, a_max=num_rows)
        c_min, c_max = np.clip([(col - offset), (col + offset + 1)], a_min=0, a_max=num_cols)

        # update mask to suppress this neighborhood
        suppressed[r_min:r_max, c_min:c_max] = True
        
    return lines


# Display the detected lines in red - overlaid on the original image
def display_lines(im: np.ndarray, lines: list) -> np.ndarray:
    """
    Overlays detected lines over original image
    :param im: original image (grayscale)
    :param lines: list of (r, theta, num_points) tuples
    :return: image annotated with detected lines
    """

    # create a color copy to draw red lines on
    if len(im.shape) == 2:
        output_img = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    else:
        output_img = im.copy()
        
    # stretch factor (larger than image diagonal)
    img_h, img_w = im.shape[:2]
    stretch = int(np.ceil(np.sqrt(img_h**2 + img_w**2))) 
    
    # draw each line - find 2 points along the line, connect them with cv2.line
    for r, theta, votes in lines:
        theta_rad = np.deg2rad(theta)
        
        # anchor point
        anchor_x = np.cos(theta_rad)
        anchor_y = np.sin(theta_rad)
        x0 = anchor_x * r
        y0 = anchor_y * r
        
        # stretch in both directions along the perpendicular
        x1 = int(x0 + stretch * (-anchor_y))
        y1 = int(y0 + stretch * (anchor_x))
        x2 = int(x0 - stretch * (-anchor_y))
        y2 = int(y0 - stretch * (anchor_x))
        
        # draw
        cv2.line(output_img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2)

    return output_img
    

# Now use the above functions to implement
def straight_lines(image_file: str | Path, resolution_r: int = 1, resolution_ang: int = 1, min_num_points: int = 50, display: bool = False, canny_th1=250, canny_th2=500, canny_aperture=5, nms_neighb_size=5) -> tuple[list, np.ndarray]:
    """
    Detects straight lines in an image using Hough Transform
    """
    # load image
    img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_file}")
        
    # extract edges using Canny
    edges = cv2.Canny(img, threshold1=canny_th1, threshold2=canny_th2, apertureSize=canny_aperture)
    
    # extract edge point coords
    y_idxs, x_idxs = np.where(edges > 0)
    edge_points = np.column_stack((x_idxs, y_idxs))
    
    # compute H matrix
    H, theta, r_ticks = H_matrix(
        L_points=edge_points, 
        resolution_r=resolution_r, 
        resolution_ang=resolution_ang, 
        img_shape=img.shape
    )  

    # extract lines from H matrix
    lines = list_lines(
        H=H, 
        min_num_points=min_num_points, 
        theta_ticks=theta, 
        r_ticks=r_ticks,
        nms_neighb_size=nms_neighb_size
    )                             
    
    # display
    annotated_img = display_lines(im=img, lines=lines)
    if display:
        plt.rcParams['figure.figsize'] = (15.0, 5.0)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img, cmap='gray'); ax1.set_title('Original Image'); ax1.axis('off')
        ax2.imshow(edges, cmap='gray'); ax2.set_title('Edges (Canny)'); ax2.axis('off')
        ax3.imshow(annotated_img); ax3.set_title(f'Detected Lines (Count: {len(lines)})'); ax3.axis('off')
        plt.show()
        
    return lines, annotated_img


# ------------- TEST SECTION A -------------
# task A.1: synthetic image
synthetic_img = np.zeros((200, 200), dtype=np.uint8)
cv2.line(synthetic_img, pt1=(20, 20), pt2=(180, 180), color=255, thickness=2)
cv2.line(synthetic_img, pt1=(20, 150), pt2=(150, 20), color=255, thickness=2)
cv2.imwrite('images/synthetic.jpg', synthetic_img)

print("--- Synthetic Image ---")
lines_synth, _ = straight_lines(
    image_file=Path('images/synthetic.jpg'), 
    resolution_r=1,
    resolution_ang=1,
    min_num_points=80,
    nms_neighb_size=31,
    display=True
)

# tasks A.2, A.4: apply to real images
print("\\n--- Crosswalk ---")
lines_crosswalk, _ = straight_lines(
    image_file=Path("images/Crosswalk.jpg"), 
    resolution_r=2, 
    resolution_ang=2, 
    min_num_points=100, 
    canny_th1=100, 
    canny_th2=200, 
    canny_aperture=3, 
    nms_neighb_size=31,
    display=True
)

print("\\n--- Lines on the Road ---")
lines_road, _ = straight_lines(
    image_file=Path("images/linesOnTheRoadGray.jpg"), 
    resolution_r=2, 
    resolution_ang=2, 
    min_num_points=120, 
    canny_th1=100, 
    canny_th2=200, 
    canny_aperture=3, 
    nms_neighb_size=31,
    display=True
)

print("\\n--- Sudoku ---")
# Using a threshold of 50 to answer Question 4
lines_sudoku, lines_on_sudoku_overlay = straight_lines(
    image_file=Path("images/Sudoku.PNG"), 
    resolution_r=2, 
    resolution_ang=1, 
    min_num_points=50, 
    nms_neighb_size=31,
    display=True
)
print(f"Number of straight lines found with >50 points in Sudoku: {len(lines_sudoku)}")

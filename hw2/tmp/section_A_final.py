# ## <span style="color:blue">Section A: Detect Straight Lines </span>
# 
# In this part you will use the set of edge points to detect straight lines in an image.\
# The input will consist of edge points computed by the Canny edge detector - you can use the implementation of CV2, which is demonstrated below.\
# The output will be a set of straight lines in the image. There are two main methods to compute straight lines from such input: the Hough transform and RANSAC. You will implement the Hough transform.
# 
# **Hough transform**\
# Every 2D line, $\ell$, can be represented by 2 parameters: $r$ and $\theta$ where all points on the line satisfy $r= x\cos\theta + y\sin\theta$.
# Let $P_0=(x_0,y_0)$ be the intersection of a normal to $\ell$ from the origin.
# The distance between $P_0$ and the origin is given by $r$ and the angle between the normal and the $x$ axis is given by $\theta$.
# 
# 

# **Your goal:**     Write the following function\
# `straight_lines(image_file, resolution_r, resolution_ang, min_num_points, display, ...)`\
# You may add other parameters; make sure to give them default values.
#     
# 

# To do so, you need to also define the following functions. 
# You may add parameters to the functions, as long as you provide clear explanations of their purpose.

# Input: a set of edge points (or corners), and the resolution of the distance and angles.
# Output: the Hough matrix (H) containing votes for lines represented by r and theta.

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
def display_lines(im: np.ndarray | None, lines: list) -> np.ndarray:
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
    

def plot_hough_lines_procedure(img: np.ndarray, edges: np.ndarray, annotated_img: np.ndarray, num_lines: int, image_name: str) -> None:
    """
    Helper function to display Hough lines process
    - original image
    - Canny edges
    - detected lines overlaid over image
    """
    plt.rcParams['figure.figsize'] = (15.0, 5.0)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.suptitle(f"Hough Transform Line Detection - {image_name}", fontsize=16)
    
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(edges, cmap='gray')
    ax2.set_title('Edges (Canny)')
    ax2.axis('off')
    
    ax3.imshow(annotated_img)
    ax3.set_title(f'Detected Lines (Count: {num_lines})')
    ax3.axis('off')
    
    plt.show()


# Now use the above functions to implement
def straight_lines(image_file: str | Path, resolution_r: int = 1, resolution_ang: int = 1, min_num_points: int = 50, display: bool = False, canny_th1=250, canny_th2=500, canny_aperture=3, nms_neighb_size=5) -> tuple[list, np.ndarray]:
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
    annotated_img = display_lines(im=img, lines=lines)

    if display:
        img_name = Path(image_file).name
        plot_hough_lines_procedure(img=img, edges=edges, annotated_img=annotated_img, num_lines=len(lines), image_name=img_name)
        
    return lines, annotated_img

# Here is an example of how to draw a red line
# between (x1, y1) and (x2, y2) on a gray level image, img
# The first step is to create a color image from img.

img = np.random.randint(0, 256, size=(50, 50), dtype=np.uint8)
x1, y1 = 10, 20
x2, y2 = 15, 40
thickness = 2

color_image = np.dstack([img, img, img])
cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), thickness)

plt.imshow(color_image)
plt.axis('off')
plt.show()

# Here is an example of how to use the CV2 Canny edge detector.
# You can play with the parameters to achieve desired results.

img = cv2.imread(str(Path("images/Sudoku.PNG")), cv2.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

edges = cv2.Canny(img, 250, 500, 5)

plt.rcParams['figure.figsize'] = (16.0, 16.0)
f, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row')

ax1.imshow(img), ax1.set_title('Original Image')
ax2.imshow(edges), ax2.set_title('Edge Image')


# **Apply and answer**
# 
# 1. Generate a synthetic image to test your Hough Transform algorithm. \
#    Submit the image as well as the results.
# 
# 2. Apply your algorithm to the following images: Crosswalk, linesOnTheRoadGray, Sudoku. \
#    Choose an appropriate set of parameters; you might need different parameters for each image. Don't forget to display the results.
#    
# 3. **Answer**: How the resolutions of $r$ and $\theta$ affect the results? Display images that demonstarte your answer.
#    
# 4. Choose one image and **answer**: how many straight lines did you find with more than 50 points? Display these lines on the image.
#    
# 5. Suggest an algorithm to compute the length of lines in the image. Describe the algorithm without implementing it.
#    
# 6. Suggest three applications that use the results of straight line detection in an image.
# 

# #### Task A1 - Synthetic Image

synthetic_img = np.zeros((200, 200), dtype=np.uint8)
cv2.line(synthetic_img, pt1=(20, 20), pt2=(180, 180), color=255, thickness=2)
cv2.line(synthetic_img, pt1=(20, 150), pt2=(150, 20), color=255, thickness=2)
cv2.imwrite('images/synthetic.jpg', synthetic_img)

lines_synth, lines_on_synthetic_overlay = straight_lines(
    image_file=Path('images/synthetic.jpg'), 
    resolution_r=1,
    resolution_ang=1,
    min_num_points=80,
    nms_neighb_size=9,
    display=True
)

# #### Task A2 - Real Images

lines_crosswalk, lines_on_crosswalk_overlay = straight_lines(
    image_file=Path("images/Crosswalk.jpg"), 
    resolution_r=2, 
    resolution_ang=2, 
    min_num_points=300, 
    canny_th1=150, # higher canny thresholds to reduce noise from asphalt textures
    canny_th2=300, 
    canny_aperture=3, 
    nms_neighb_size=15,
    display=True
)

lines_road, lines_on_road_overlay = straight_lines(
    image_file=Path("images/linesOnTheRoadGray.jpg"), 
    resolution_r=2, 
    resolution_ang=2, 
    min_num_points=200, 
    canny_th1=100, 
    canny_th2=200, 
    canny_aperture=3, 
    nms_neighb_size=15,
    display=True
)

lines_sudoku, lines_on_sudoku_overlay = straight_lines(
    image_file=Path("images/Sudoku.PNG"), 
    resolution_r=2, 
    resolution_ang=2, 
    min_num_points=250, 
    nms_neighb_size=15,
    canny_th1=100,
    canny_th2=200,
    display=True
)

# #### Task A3 - Effect of Resolution Parameters

# images + neutral parameters to isolate resolution effect
test_images = [
    {"path": "images/synthetic.jpg", "t1": 100, "t2": 200, "vote_threshold": 80, "nms": 11},
    {"path": "images/Crosswalk.jpg", "t1": 250, "t2": 500, "vote_threshold": 150, "nms": 15},
]

# test configurations
test_resolutions = [
    {"r": 1, "ang": 1, "title": "BASELINE (r=1, ang=1)"},
    {"r": 15, "ang": 1, "title": "COARSE R (r=15, ang=1)"},
    {"r": 1, "ang": 15, "title": "COARSE THETA (r=1, ang=15)"}
]

for img_dict in test_images:

    # load image
    img_path = Path(img_dict["path"])
    if not img_path.exists():
        print(f"Skipping {img_path.name} - File not found.")
        continue
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        
    # extract edge points
    edges = cv2.Canny(img, threshold1=img_dict["t1"], threshold2=img_dict["t2"], apertureSize=3)
    y_idxs, x_idxs = np.where(edges > 0)
    L_points = np.column_stack((x_idxs, y_idxs))
    
    # grid for image visuals
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    fig.suptitle(f"Resolution Test: {img_path.name}", fontsize=16, fontweight='bold', y=0.98)
    
    for i, res_dict in enumerate(test_resolutions):
        res_r = res_dict["r"]
        res_ang = res_dict["ang"]
        config_name = res_dict["title"]
        
        # computte H matrix
        H, theta, r_vals = H_matrix(
            L_points=L_points, 
            resolution_r=res_r, 
            resolution_ang=res_ang, 
            img_shape=img.shape
        )
        
        # detect + draw lines
        lines = list_lines(
            H=H, 
            min_num_points=img_dict["vote_threshold"], 
            theta_ticks=theta, 
            r_ticks=r_vals, 
            nms_neighb_size=img_dict["nms"]
        )
        annotated_img = display_lines(im=img, lines=lines)
        
        # top row: H matrix
        ax_h = axes[0, i]
        ax_h.imshow(H, cmap='hot', aspect='auto')
        ax_h.set_title(f"{config_name} | Line Count: {len(lines)}", fontsize=12)
        ax_h.set_xlabel("Theta Bins"); ax_h.set_ylabel("r Bins")
        
        # bottom row: final image
        ax_img = axes[1, i]
        ax_img.imshow(annotated_img)
        ax_img.axis('off')

    plt.tight_layout()
    plt.show()


# ##### Observations:
# The resolutions of r and theta dictate the size of the Hough bins, controlling tolerance for 
# variations in position and angle.
# 
# Fine resolution on both:
# * requires points to fall within a very narrow/limited "strip" of lines
# * can distinguish closely spaced lines if they are well defined
# * struggles with thick or noisy lines (see double-detetions in crosswalk image) -> votes from single thick edge fall into multiple adjacent bins, leading to: 
#     - multiple detections of same real edge
#     - missed detections if bins fall short of vote threshold
# 
# Coarse r resolution:
# * reduces line position strictness - edge points falling on parallel lines within the broader range of r are grouped
# * groups distant, unrelated pixels into the same bin because they share an angle -> creates FP lines
# * makes TP physically shift away from their real edges
# 
# Coarse theta resolution: 
# * reduces angle strictness - points sitting on lines with similar but not identical angles are grouped
# * better with thick edges nd noisy lines - groups votes scattered onto deifferent angles close enough in same bin (fixes the double-detection issue in crosswalk image)
# * can make TP detections change their angle a bit from the real edge angle
# 

# #### Task A4 - Low Vote Threshold (on crowsswalk image)
# Number of lines detected in image `Crosswalk.jpg` with at least 50 points: 290

lines_low_vote_threshold, lines_low_vote_threshold_overlay = straight_lines(
    image_file=Path("images/Crosswalk.jpg"), 
    resolution_r=2, 
    resolution_ang=2, 
    min_num_points=50, 
    canny_th1=150,
    canny_th2=300, 
    canny_aperture=3, 
    nms_neighb_size=15,
    display=True
)

print(f"Number of straight lines found with >50 points in Crosswalk: {len(lines_low_vote_threshold)}")  

# #### Task A5 - Line Length Algorithm
# 1. For each detected line defined by $(r, \theta)$:
#     1. Get the Canny edge points that fall on that line (filter from the `L_points` array).
#     2. Find vector in direction of the line - the normal vector is $\langle cos(\theta),sin(\theta)\rangle$ so a $90\degree$ rotation is a vector in the direction of the line: $\langle -sin(\theta),cos(\theta)\rangle$.
#     3. Project each edge (2D) point onto the line - dot product of point's $\langle x,y\rangle$ and the line direction vector.
#     4. Sort the points by their 1D coord on the line.
#     5. Pick an upper threshold for the difference 2 points can have and still be part of the same line segment; cluster the points using this threshold (when the gap is too big start a new cluster)
#     6. Measure line (segment) length - each segment's length is the difference bettween the 1D coord of its first and last point.

# #### Task A6 - Line Detection Applications
# 1. Autonomous vehicle lane detection (when I worked at GM it was mostly done with deep CV, but Hough is useful for validation and visualization).
# 2. Sports analytics/boradcasting - detecting court boundaries/markings (augmenting game broadcast or calibrating camera tracking)
# 3. Image editing - finding important lines like the horizon (e.g. image leveling) or cropping important parts (e.g. document scanning)
# 

# Section A - Apply and answer (student code)

# A4
num_lines_vote_threshold_50 = len(lines_low_vote_threshold)

# A3
resolution_effect = """
The resolutions of r and theta dictate the size of the Hough bins, controlling tolerance for 
variations in position and angle.\n\n

Fine resolution on both:\n
- requires points to fall within a very narrow/limited "strip" of lines\n
- can distinguish closely spaced lines if they are well defined\n
- struggles with thick or noisy lines (see double-detetions in crosswalk image) -> votes from 
single thick edge fall into multiple adjacent bins, leading to: a) multiple detections of same 
real edge and b) missed detections if bins fall short of vote threshold\n\n

Coarse r resolution:\n
- reduces line position strictness - edge points falling on parallel lines within the broader range of r are grouped\n
- groups distant, unrelated pixels into the same bin because they share an angle -> creates FP lines\n
- makes TP physically shift away from their real edges\n\n

Coarse theta resolution:\n
* reduces angle strictness - points sitting on lines with similar but not identical angles are grouped\n
* better with thick edges nd noisy lines - groups votes scattered onto deifferent angles close enough in 
same bin (fixes the double-detection issue in crosswalk image)\n
* can make TP detections change their angle a bit from the real edge angle\n
"""

# A5
line_length_algorithm = """
1. For each detected line defined by (r, theta):\n
1.1 Get the Canny edge points that fall on that line (filter from the L_points array).\n
1.2 Find vector in direction of the line - the normal vector is [cos(theta),sin(theta)] so a 90 degree rotation is a vector in the direction of the line: [-sin(theta),cos(theta)].\n
1.3 Project each edge (2D) point onto the line - dot product of point's [x,y] and the line direction vector.\n
1.4 Sort the points by their 1D coord on the line.\n
1.5 Pick an upper threshold for the difference 2 points can have and still be part of the same line segment; cluster the points using this threshold (when the gap is too big start a new cluster).\n
1.6 Measure line (segment) length - each segment's length is the difference bettween the 1D coord of its first and last point.\n
"""

# A6
line_detection_applications = [
    "Autonomous vehicle lane detection (when I worked at GM it was mostly done with deep CV, but Hough is useful for validation and visualization)",
    "Sports analytics/boradcasting - detecting court boundaries/markings (augmenting game broadcast or calibrating camera tracking)", 
    "Image editing - finding important lines like the horizon (e.g. image leveling) or cropping important parts (e.g. document scanning)"
]

# ## Simple Submission Flow
# 
# 1. Implement your code in Section A, Section B, and Section C cells.
# 2. Define the final variables listed below.
# 3. Run `Results Display` (last cell).
# 4. Submit this notebook (results are displayed at the end).
# 
# ## Final Variables to Display
# 
# Define these variables in your notebook code. The last cell reads them directly.
# 
# **Section A – Hough Transform:**
# - `lines_on_sudoku_overlay` – colour image (H x W x 3) of Sudoku with detected lines drawn in red
# - `resolution_effect` (short text - answer to Q3)
# - `line_length_algorithm` (short text - answer to Q5)
# - `line_detection_applications` (list of 3 strings - answer to Q6)
# 
# **Section B – Patch Matching:**
# - `match_overlay` – colour image showing matches between view0 & view6 (no y-constraint)
# - `match_overlay_y` – colour image showing matches with y-coordinate constraint
# - `matching_analysis` (short text - answer to Q4)
# - `reliable_match_regions` (short text - answer to Q5)
# 
# **Section C – Stereo Depth:**
# - `disparity_map` (2D array)
# - `depth_map` (2D array)

# ## Final Results Display

import numpy as np
import matplotlib.pyplot as plt


def _show_text(name, value):
    if value is None:
        print(f"  {name}: MISSING")
    else:
        print(f"  {name}: {value}")


def _show_image(name, value, figsize=(6, 5)):
    if value is None:
        print(f"  {name}: MISSING")
        return
    arr = np.asarray(value)
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
    plt.figure(figsize=figsize)
    if arr.ndim == 2:
        plt.imshow(arr, cmap="gray")
    else:
        plt.imshow(arr)
    plt.title(name)
    plt.axis("off")
    plt.show()


# ==================== Section A ====================
print("=" * 60)
print("Section A - Hough Transform")
print("=" * 60)
_show_image("lines_on_synthetic_overlay",
            globals().get("lines_on_synthetic_overlay", None))
_show_image("lines_on_sudoku_overlay",
            globals().get("lines_on_sudoku_overlay", None))
_show_text("resolution_effect",
           globals().get("resolution_effect", None))
_show_text("# of lines with at least 50 points",
           globals().get("num_lines_vote_threshold_50", None))
_show_text("line_length_algorithm",
           globals().get("line_length_algorithm", None))
_show_text("line_detection_applications",
           globals().get("line_detection_applications", None))

# # ==================== Section B ====================
# print("\n" + "=" * 60)
# print("Section B - Patch Matching")
# print("=" * 60)
# _show_image("match_overlay",
#             globals().get("match_overlay", None), figsize=(14, 5))
# _show_image("match_overlay_y",
#             globals().get("match_overlay_y", None), figsize=(14, 5))
# _show_text("matching_analysis",
#            globals().get("matching_analysis", None))
# _show_text("reliable_match_regions",
#            globals().get("reliable_match_regions", None))

# # ==================== Section C ====================
# print("\n" + "=" * 60)
# print("Section C - Depth from Stereo")
# print("=" * 60)
# disp = globals().get("disparity_map", None)
# depth = globals().get("depth_map", None)
# if disp is not None:
#     arr = np.asarray(disp)
#     print(f"  disparity_map: shape={arr.shape}, dtype={arr.dtype}, "
#           f"min={arr.min():.1f}, max={arr.max():.1f}")
#     plt.figure(figsize=(8, 5))
#     plt.imshow(arr, cmap="jet"); plt.colorbar(label="disparity")
#     plt.title("disparity_map"); plt.axis("off"); plt.show()
# else:
#     print("  disparity_map: MISSING")
# if depth is not None:
#     arr = np.asarray(depth)
#     print(f"  depth_map: shape={arr.shape}, dtype={arr.dtype}, "
#           f"min={arr.min():.1f}, max={arr.max():.1f}")
#     plt.figure(figsize=(8, 5))
#     plt.imshow(arr, cmap="jet"); plt.colorbar(label="depth (mm)")
#     plt.title("depth_map"); plt.axis("off"); plt.show()
# else:
#     print("  depth_map: MISSING")

# Plan: hw2 Computer Vision Notebook

## Overview
Three sections: (A) Hough Transform line detection, (B) Patch matching, (C) Stereo depth. Only NumPy + Python stdlib + provided imports allowed. Final cell reads named variables for grading.

---

## Section A: Hough Transform

### Functions to implement
1. `H_matrix(L_points, resolution_r, resolution_ang)` — builds accumulator
2. `list_lines(H, min_num_points)` — returns list of (r, θ, num_points) triplets
3. `display_lines(im, lines)` — draws lines in red on grayscale image
4. `straight_lines(image_file, resolution_r, resolution_ang, min_num_points, display)` — orchestrates pipeline

### Apply & Answer tasks
- Q1: Generate synthetic image to test; show image + results
- Q2: Apply to Crosswalk.jpg, linesOnTheRoadGray.jpg, Sudoku.PNG
- Q3: Answer + demo images: how r and θ resolution affect results
- Q4: Count lines with >50 points in one image; display them
- Q5: Suggest (text only) algorithm for line length computation
- Q6: List 3 applications of straight line detection

### Final variables
- `lines_on_sudoku_overlay` (H×W×3 uint8)
- `resolution_effect` (string)
- `line_length_algorithm` (string)
- `line_detection_applications` (list of 3 strings)

---

## Section B: Patch Matching

### Functions to implement
1. `SSD(patch_descr_1, patch_descr_2)` — scalar or vectorized
2. `NCC(patch_descr_1, patch_descr_2)` — scalar or vectorized
3. `patch_from_im(im, p, size)` — raw pixel vector
4. `hist_patch_im(im, p, size)` — 30-bin grey histogram
5. `gradient(im, p, size)` — gradient strength vector
6. `hist_gradient(im, p, size)` — 30-bin gradient histogram

### Apply & Answer tasks
- Q1: Compute Harris corners on view0.tif + view6.tif
- Q2A: Up to 2000 strongest corners, fixed-scale patch, match + display with lines
- Q2B: Up to 4000 corners with y-coordinate constraint (rectified pair)
- Q2C: Ratio test (best/second-best); demonstrate effectiveness
- Q3: Compare descriptors + SSD vs NCC with examples
- Q4: Identify and mark incorrect matches for A, B, C; answer which case has more errors
- Q5: Where are reliable matches in the scene?

### Final variables
- `match_overlay` (H×(W1+W2)×3)
- `match_overlay_y` (same format)
- `matching_analysis` (string)
- `reliable_match_regions` (string)

---

## Section C: Stereo Depth

### Functions to implement
1. Dense matching function — NCC similarity along same-y rows, sliding window (sx, sy), disparity range (d_min, d_max)
2. Disparity map function — returns D matrix
3. Depth computation — Z = α·baseline / disparity (+100 offset); X, Y from pixel coords
4. 3D plot of X, Y, Z

### Parameters
- Images: view1.tif (left), view5.tif (right)
- α_x = α_y = 1, baseline = 160mm
- Add 100 to disparity when computing depth

### Final variables
- `disparity_map` (H×W float)
- `depth_map` (H×W float)

---

## Recommended Implementation Order
Phase 1: Section A core functions → test on synthetic image → apply to 3 real images
Phase 2: Section B descriptors + SSD/NCC scalar → vectorize → matching experiments
Phase 3: Section C dense matching → disparity → depth → 3D plot
Phase 4: Written answers for all text variables
Phase 5: Verify final display cell runs end-to-end

## Key constraints
- No imports beyond cv2, matplotlib, numpy, scipy.linalg.null_space
- Vectorized code required
- All 10 final variables must be defined before last cell

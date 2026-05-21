# Simple — pupil detector

Threshold-based detector. Pixels below `pupil_threshold` form the
candidate mask, `cv::findContours` walks the dark regions, opt-in
shape-quality gates (`min_ellipse_fit_ratio`, `min_roundness_ratio`)
pick the first contour that looks pupil-shaped, `cv::fitEllipse` runs
on its convex hull, and the pupil centre comes from one of five
methods (default: periodic interpolating cubic spline through the
convex hull, with Green's-theorem centroid of the enclosed area).

MIT-licensed (covered by the top-level [LICENSE](../../../../LICENSE)).

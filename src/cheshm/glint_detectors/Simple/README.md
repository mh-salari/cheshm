# Simple — glint detector

Threshold-based detector. Bright pixels inside a search disk around
the pupil centre form the candidate region, contours are filtered by
area, half-plane (`keep_above` / `keep_below` / `keep_left` /
`keep_right`) and shape-quality gates, optionally split when the rig
has one more LED than contours found, sorted left-to-right, and each
glint's centre is computed via one of five methods.

MIT-licensed (covered by the top-level [LICENSE](../../../../LICENSE)).

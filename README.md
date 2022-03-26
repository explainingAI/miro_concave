# Improving concave point detection to better segment overlapped objects in images
## Authors: M. Miró-Nicolau, G. Moyà-Alcover, M. González-Hidalgo, A. Jaume-i-Capó

### Abstract of the paper
This paper presents a method that improve state-of-the-art of the concave point detection
methods as a first step to segment overlapping objects on images. It is based on the analysis
of the curvature of the objects’ contour. The method has three main steps. First, we
pre-process the original image to obtain the value of the curvature on each contour point.
Second, we select regions with higher curvature and we apply a recursive algorithm to refine
the previous selected regions. Finally, we obtain a concave point from each region based on
the analysis of the relative position of their neighbourhood
We experimentally demonstrated that a better concave points detection implies a better
cluster division. In order to evaluate the quality of the concave point detection algorithm,
we constructed a synthetic dataset to simulate overlapping objects, providing the position
of the concave points as a ground truth. As a case study, the performance of a well-known
application is evaluated, such as the splitting of overlapped cells in images of peripheral
blood smears samples of patients with sickle cell anaemia. We used the proposed method
to detect the concave points in clusters of cells and then we separate this clusters by ellipse
fitting.

### Reference

**UNDER REVISION**
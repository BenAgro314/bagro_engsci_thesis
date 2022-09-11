Sim design:

Create a bunch of example instances in the following way:

0. (Randomly) Initialize the camera parameters
1. Randomly (how to do this correctly?) initialize the pose of the camera
2. In the camera frame, randomly add N points in its FOV. Convert those coordinates to the world frame.
3. Use the generative model to project the points bi-linearly onto the left and right camera images 

Outputs: Points in world frame $p_w_k$, measurements in camera frame $y_k$, ground truth camera pose, camera parameter matrix $M$; visualizations of the aformentioned outputs
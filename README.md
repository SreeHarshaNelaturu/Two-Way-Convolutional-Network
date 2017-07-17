# TwoWayConvNet

Contributors: Sree Harsha Nelaturu, Rohan Pooniwala, Sourav Sharan and Aparna Krishnakumar

Implementation of the research paper  "Two-Stream Convolutional Networks for Action Recognition in Videos " https://arxiv.org/pdf/1406.2199.pdf 

Approach:
The approach involves building and training 2 convolutional neural networks simultaneously- one spatial network which captures action recognition on still images ina video frame by frame and a temproal network which computes motion between frames in a video by stacking optical flow field vectors on top of one another. The output of both the convolutional networks are then passed through a SVM to detect action in videos. 

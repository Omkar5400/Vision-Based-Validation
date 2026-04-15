Vision-Based Validation of Passive Safety Deployments.

Project Overview

This repository contains an end-to-end AI-supported tool designed to automate the validation of front restraint and airbag deployment simulations against high-speed crash video data. The system synchronizes high-frequency simulation telemetry with unstructured technical video data to identify anomalies and ensure simulation accuracy.

Key Features:

* Automated Airbag Analysis: Uses a custom PyTorch and OpenCV pipeline to analyze deployment kinetics, such as expansion rates and trigger timing, from crash videos.
* Advanced Computer Vision: Implements Semantic Segmentation (U-Net) and Optical Flow for precise fabric tracking, featuring a custom Luminance Gate to handle complex cabin lighting and reflections.
* Metric Derivation: Formulates custom evaluation algorithms to extract physical indicators like Time-to-Peak (TTP) and Geometric Ratios.
* LS-DYNA Correlation: Features a synchronization workflow that compares vision-extracted data against explicit FEA (LS-DYNA) results, achieving a 98% Pearson correlation.
  
Architecture & Methodology:

The tool follows a modular multimodal AI pipeline architecture:

1. Preprocessing: High-speed video frame extraction and image-based feature enhancement.
2. Segmentation Layer: Pixel-level isolation of deployment zones using Deep Learning.
3. Kinematic Extraction: Vectorized calculations for velocity and displacement.
4. Statistical Validation: Correlation of vision metrics with simulation ground truth.
   
Technical Stack:

* Programming: Python (NumPy, Pandas, SciPy).
* AI/DL: PyTorch, TensorFlow, OpenCV.
* Simulation: LS-DYNA (Validation/Correlation).
* Environment: Developed as a scalable prototype for industrial automotive V-Model workflows.

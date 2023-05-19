# linearlayers
Numerical experiments for adding linear layers to shallow ReLU neural networks

These experiments were run on Google Colab using a T4 GPU. 

The models are trained in `teacher_networks_linear_layers.ipynb`. This file saves the trained models and training-time logs to the files currently found in `TeacherNetworkResults`.
Running this notebook takes approximately 1 hour.

Plots and tables with information about the active subspace, generalization, and out-of-distribution performance of trained models are created in `teacher_networks_plotting_results.ipynb`. 

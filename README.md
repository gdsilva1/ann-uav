# ANN applied to UAV

This repositpry contains a research project that aims to apply artifical neural netowrks (ANN) in the control system field.

## 1. Main Goal

Literature has not much content using ANNs in the control system applied to unnamed aerial vehicles (UAv). The goal is to build a model that can predict the control forces based on the trajectory of an UAV.

## 2. Methodolody

With a small sample of a forces-trajectory set, a simple regression ANN model was built using TensorFlow and then migrated to PyTorch.

## 3. Results

THe results are rudimentar but shows good perspective, given the limitations of the implementation.

### 3.1 Limitations

The samples were generated using MATLAB, as the control model was implemented within the software. This way, the mathematical control model was not optimized to generete multiple trajectories, thus to generate the sample, a powerful computer processor is mandatory. An alternative is to reimplement the mathematical model considering the the memory and processor use. To this project, this is out of scope.

To traing the ANN, it was used 1000 samples, being 800 to training and 200 to testing, which is a small sample for training.

Since it started with the simplest model possible (regression ANN), the model not necessary performs well to the predicted data.

## 4. Next Steps

TBD

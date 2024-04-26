# ANN applied to UAV

This repositpry contains a research project that aims to apply artifical neural netowrks (ANN) in the control system field.

## 1. Main Goal

Literature has not much content using ANNs in the control system applied to unnamed aerial vehicles (UAv). The goal is to build a model that can predict the control forces based on the trajectory of an UAV.

## 2. Methodolody

With a small sample of a forces-trajectory set, a simple regression ANN model was built using TensorFlow and then migrated to PyTorch.

## 3. Results

THe results are rudimentar but shows good potential, given the limitations of the implementation.

### 3.1 Limitations

The samples were generated using MATLAB, as the control model was implemented within the software. Thus, the mathematical control model was not optimized to generete multiple trajectories. THerefore, to generate the sample, a powerful computer processor is mandatory. An alternative is to reimplement the mathematical model considering the the memory and processor use. However, this is out of scope for this project.

To traing the ANNs, 1000 were used, with 800 for training and 200 for testing, which is a small sample for training.

Since it started with the simplest model possible (regression ANN), the model may not necessarily perform well on the predicted data.

## 4. Next Steps

Some suggestions to improve the model:

- get a bigger sample;
- build a more sophisticated ANN architecture;
- split different segments of the trajectory among different ANNs;
- utilize the forces predicted by the ANNs in the white box script and verify if the flight is executed appropriately;
- use different out-of-scope deep learning techniques and try traditional machine learning techniques for well-structured data.

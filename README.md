[TensorFlow 2] Variational Inference with Normalizing Flows
=====
TensorFlow implementation of "Variational Inference with Normalizing Flows" [1]

## Concept
<div align="center">
  <img src="./figures/flow.png" width="550">  
  <p>Concept of the Normalizing Flow (NF).</p>
</div>

<div align="center">
  <img src="./figures/algorithm.png" width="550">  
  <p>Algorithm for training NF.</p>
</div>

## Results

<div align="center">
  <img src="./figures/reconstruction.gif" width="800">
  <p><strong>Upper</strong>: target image; <strong>Lower</strong>: restored image.</p>
</div>

<div align="center">
  <img src="./figures/energy.gif" width="600">  
  <p><strong>Left</strong>: first density z_0; <strong>Right</strong>: last density z_k.</p>
</div>

## Requirements
* Python 3.7.6  
* Tensorflow 2.3.0  
* Numpy 1.18.15
* whiteboxlayer 0.1.17

## Reference
[1] Danilo Jimenez Rezende and Shakir Mohamed. (2015). <a href="https://arxiv.org/abs/1505.05770">Variational Inference with Normalizing Flows</a>. arXiv preprint arXiv:1505.05770.

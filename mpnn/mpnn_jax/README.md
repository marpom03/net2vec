
# Introduction

This repository contains a JAX re-implementation of the Message-Passing Neural Network model introduced in K. Rusek, P. Chołda, *“Message-Passing Neural Networks Learn Little’s Law”*. The code is based on the authors’ original TensorFlow 1 implementation and was developed as part of the engineering thesis **“Reproducibility of results from paper Message-Passing Neural Networks Learn Little's Law implemented in JAX”** supervised by dr inż. Krzysztof Rusek.

The JAX implementation is provided in two APIs: **Flax Linen** and **Flax NNX**, allowing a direct comparison of modern JAX frameworks.

Below we present instructions on how to run our code and compare the results obtained by the original TensorFlow 1 model with the JAX implementations.

# How to run
Below you can find a detailed instruction on how to prepare the data, train the model, and test it.

## Generating samples

First, you need samples to train the model. To generate them, use the samples.py script, which creates three sets of samples:

-   training set

-   validation set (used during training)
    
-   test set (used after training)
    

The configuration for sample generation is located in the config.py file, with the following settings:

-   **output_dir** - directory where the generated samples will be saved
    
-   **seed** - RNG seed used during sample generation
    
-   **n_nodes** - number of nodes in the generated graphs
    
-   **train_size** — number of samples in the training set
    
-   **val_size** - number of samples in the validation set
    
-   **test_size** - number of samples in the test set
    
-   **rl** - lower bound for traffic load generation
    
-   **rh** - upper bound for traffic load generation
    
-   **graph_type** - graph type (BarabasiAlbert or ErdosRenyi)
    
-   **snd_path** - optional XML path to an SNDlib network, if samples should be generated from real topologies
    

## Training the model

Training is configured through the TrainingConfig class in config.py. The key parameters are:

-   **train_dataset_path** - directory containing the training samples
    
-   **val_dataset_path** - directory containing the validation samples
    
-   **output_path** - directory where the trained model will be saved
    
-   **batch_size** - number of graphs per batch
    
-   **learning_rate** - learning rate used during optimization
    
-   **seed** - RNG seed for training
    
-   **steps** - number of training steps
    
-   **log_interval** - how often validation is performed
    
-   **use_early_stopping** - whether to enable early stopping
    
-   **early_stopping_patience** - patience parameter for early stopping
    
-   **norm_profile** - normalization settings for μ and W depending on the graph model
    

After setting all training parameters, run one of the training scripts:

```
python train_linen.py
```

or
```
python train_nnx.py
```
  

## Testing the model

Before testing, configure the TestConfig section in config.py:

-   **test_dataset_path** - directory containing the test samples  


-   **checkpoint_path** - directory where the trained model is stored  
      
    
-   **output_path** - directory where test results will be saved  
      
    
-   **norm_profile** - same normalization settings used during training  



Run the desired test script:

```
python test_linen.py
```
Or
```
python test_nnx.py
```
  
## Test output

After testing, two plots will be saved in output_path:

### **Predicted vs. True Values**  

   A scatter plot showing how close the predicted labels are to the true labels.  
    Points aligned along the diagonal indicate accurate predictions.  
      
    
### **Histogram of Residuals**  
   A histogram showing the distribution of prediction errors (predicted - true).  
    A narrow peak near zero indicates low residuals and good model performance.
    
# Evaluation Results

Below we present the evaluation results for both JAX implementations, as well as TensorFlow 1 tested on multiple datasets, including synthetic random graphs and real topologies from **SNDlib**.

Each evaluation reports:

- **MSE**
- **R²**
- **Pearson correlation**

---

### BA / BA

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.007 |  0.993 |  0.997 |
| **JAX Linen**    |  0.005 |  0.995 |  0.998 |
| **JAX NNX**      |  0.005 |  0.994 |  0.998 |


<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_ba/ba-ba1_white.png" width="450"></td>
    <td><img src="plots/linen/ba_ba/eval.svg" width="450"></td>
    <td><img src="plots/nnx/ba_ba/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_ba/ba-ba2_white.png" width="450"></td>
    <td><img src="plots/linen/ba_ba/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/ba_ba/residuals_hist.svg" width="450"></td>
  </tr>
</table>



---

### BA / ER

| Model            |    MSE |       R² | Pearson |
| :--------------- | -----: | -------: | ------: |
| **TensorFlow 1** |  11.50 |   -20.22 |  0.863 |
| **JAX Linen**    |   4.876 |   -8.237 | 0.778 |
| **JAX NNX**      |  10.544 |  -18.973 | 0.806 |
<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_er/ba-er1_white.png" width="450"></td>
    <td><img src="plots/linen/ba_er/eval.svg" width="450"></td>
    <td><img src="plots/nnx/ba_er/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_er/ba-er2_white.png" width="450"></td>
    <td><img src="plots/linen/ba_er/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/ba_er/residuals_hist.svg" width="450"></td>
  </tr>
</table>

---

### BA / germany50

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | -------: | ------: |
| **TensorFlow 1** |   2.248 |   -4.774 |  0.943 |
| **JAX Linen**    |   0.607 |  -0.572 | 0.410 |
| **JAX NNX**      |   2.157 |   -4.586 | 0.883 |

<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_germany50/ba-germany501_white.png" width="450"></td>
    <td><img src="plots/linen/ba_germany50/eval.svg" width="450"></td>
    <td><img src="plots/nnx/ba_germany50/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_germany50/ba-germany502_white.png" width="450"></td>
    <td><img src="plots/linen/ba_germany50/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/ba_germany50/residuals_hist.svg" width="450"></td>
  </tr>
</table>


---

### BA / SNDlib

| Model            |     MSE |       R² | Pearson |
| :--------------- | ------: | -------: | ------: |
| **TensorFlow 1** |   1.495 |   -2.363 |  0.512 |
| **JAX Linen**    |   0.720 |  -2.565 | 0.981 |
| **JAX NNX**      |   1.902 |  -2.975 | 0.466 |


<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_sndlib/ba-sndlib1_white.png" width="450"></td>
    <td><img src="plots/linen/ba_sndlib/eval.svg" width="450"></td>
    <td><img src="plots/nnx/ba_sndlib/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/ba_sndlib/ba-sndlib2_white.png" width="450"></td>
    <td><img src="plots/linen/ba_sndlib/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/ba_sndlib/residuals_hist.svg" width="450"></td>
  </tr>
</table>

---

### ER / ER

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.019 |  0.982 |  0.994 |
| **JAX Linen**    |  0.013 |  0.987 |  0.994 |
| **JAX NNX**      |  0.014 |  0.986 |  0.994 |



<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_er/er-er1_white.png" width="450"></td>
    <td><img src="plots/linen/er_er/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_er/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_er/er-er2_white.png" width="450"></td>
    <td><img src="plots/linen/er_er/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_er/residuals_hist.svg" width="450"></td>
  </tr>
</table>

---

### ER / BA

| Model            |    MSE |     R² | Pearson |
| :--------------- | -----: | -----: | ------: |
| **TensorFlow 1** | 0.116 |  0.937 | 0.977 |
| **JAX Linen**    | 0.229 | 0.876 | 0.942 |
| **JAX NNX**      | 0.755 | 0.590 | 0.896 |



<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_ba/er-ba1_white.png" width="250"></td>
    <td><img src="plots/linen/er_ba/eval.svg" width="250"></td>
    <td><img src="plots/nnx/er_ba/eval.svg" width="250"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_ba/er-ba2_white.png" width="250"></td>
    <td><img src="plots/linen/er_ba/residuals_hist.svg" width="250"></td>
    <td><img src="plots/nnx/er_ba/residuals_hist.svg" width="250"></td>
  </tr>
</table>


---

### ER / ER60

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.115  |  0.924  |  0.972 |
| **JAX Linen**    |  0.065 |  0.957 |  0.980 |
| **JAX NNX**      |  0.081 |  0.946 |  0.976 |


<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_er60/er-er601_white.png" width="450"></td>
    <td><img src="plots/linen/er_er60/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_er60/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_er60/er-er602_white.png" width="450"></td>
    <td><img src="plots/linen/er_er60/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_er60/residuals_hist.svg" width="450"></td>
  </tr>
</table>


---

### ER / janos_us

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.021 |  0.947 |  0.989 |
| **JAX Linen**    |  0.012 | 0.969 | 0.987 |
| **JAX NNX**      |  0.011 | 0.971 | 0.990 |

<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_janos-us/er-janos-us1_white.png" width="450"></td>
    <td><img src="plots/linen/er_janos-us/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_janos-us/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_janos-us/er-janos-us2_white.png" width="450"></td>
    <td><img src="plots/linen/er_janos-us/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_janos-us/residuals_hist.svg" width="450"></td>
  </tr>
</table>


---

### ER / germany50

| Model            |     MSE |     R² | Pearson |
| :--------------- | ------: | -----: | ------: |
| **TensorFlow 1** |  0.195 |  0.737 | 0.953 |
| **JAX Linen**    |  0.106 |  0.855 | 0.953 |
| **JAX NNX**      |  0.489 |  0.334 | 0.882 |




<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_germany50/er-germany501_white.png" width="450"></td>
    <td><img src="plots/linen/er_germany50/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_germany50/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_germany50/er-germany502_white.png" width="450"></td>
    <td><img src="plots/linen/er_germany50/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_germany50/residuals_hist.svg" width="450"></td>
  </tr>
</table>


---

### ER / cost266

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.035 |  0.936 | 0.987 |
| **JAX Linen**    |  0.017 | 0.970 | 0.985 |
| **JAX NNX**      |  0.015 | 0.973 | 0.988 |

<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_cost266/er-cost2661_white.png" width="450"></td>
    <td><img src="plots/linen/er_cost266/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_cost266/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_cost266/er-cost2662_white.png" width="450"></td>
    <td><img src="plots/linen/er_cost266/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_cost266/residuals_hist.svg" width="450"></td>
  </tr>
</table>


---

### ER / janos_us_ca

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.043 | 0.926 | 0.985 |
| **JAX Linen**    |  0.023 | 0.959 | 0.982 |
| **JAX NNX**      |  0.019 | 0.966 | 0.984 |


<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_janos-us-ca/er-janos-us-ca1_white.png" width="450"></td>
    <td><img src="plots/linen/er_janos-us-ca/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_janos-us-ca/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_janos-us-ca/er-janos-us-ca2_white.png" width="450"></td>
    <td><img src="plots/linen/er_janos-us-ca/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_janos-us-ca/residuals_hist.svg" width="450"></td>
  </tr>
</table>

---

### ER / SNDlib

| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.073 | 0.914 | 0.978 |
| **JAX Linen**    |  0.053 | 0.941 | 0.972 |
| **JAX NNX**      |  0.052 | 0.943 | 0.977 |


<table>
  <tr>
    <th style="text-align:center;">TensorFlow 1</th>
    <th style="text-align:center;">JAX Linen</th>
    <th style="text-align:center;">JAX NNX</th>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_sndlib/er-sndlib1_white.png" width="450"></td>
    <td><img src="plots/linen/er_sndlib/eval.svg" width="450"></td>
    <td><img src="plots/nnx/er_sndlib/eval.svg" width="450"></td>
  </tr>
  <tr>
    <td><img src="plots/tf1/er_sndlib/er-sndlib2_white.png" width="450"></td>
    <td><img src="plots/linen/er_sndlib/residuals_hist.svg" width="450"></td>
    <td><img src="plots/nnx/er_sndlib/residuals_hist.svg" width="450"></td>
  </tr>
</table>

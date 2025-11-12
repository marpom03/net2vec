# Comparison of MPNN Implementations


The following tables present a comparison between three implementations of the Message Passing Neural Network (MPNN):
the original TensorFlow 1 version (as proposed in Message-Passing Neural Networks Learn Little’s Law), and two modern re-implementations in JAX - Linen and JAX - NNX frameworks.

**Training setup:** All models were trained for **200,000 iterations** on **20,000 training samples** with **200 validation samples**; each test set contained **2,000 samples**.

### BA / BA


| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.0064 |  0.9923 |  0.9972 |
| **JAX Linen**    | 0.0045 | 0.9953 | 0.9978 |
| **JAX NNX**      | 0.0158 | 0.9837 | 0.9967 |


---


### BA / ER


| Model            |    MSE |     R² | Pearson |
| :--------------- | -----: | -----: | ------: |
| **TensorFlow 1** |  11.62 |  -18.56 |   0.849 |
| **JAX Linen**    |  4.987 |  −8.45 |   0.790 |
| **JAX NNX**      | 10.84 | −19.53 |   0.846 |


---








### ER / ER


| Model            |     MSE |      R² | Pearson |
| :--------------- | ------: | ------: | ------: |
| **TensorFlow 1** |  0.0204 |  0.9802 |  0.9937 |
| **JAX Linen**    | 0.0129 | 0.9872 | 0.9939 |
| **JAX NNX**      | 0.0179 | 0.9821 | 0.9946 |


---


### ER / BA


| Model            |   MSE |     R² | Pearson |
| :--------------- | ----: | -----: | ------: |
| **TensorFlow 1** | 0.125 |  0.933 |   0.975 |
| **JAX Linen**    | 0.229 | 0.876 |   0.942 |
| **JAX NNX**      | 0.124 | 0.933 |   0.973 |


---


### ER / ER60


| Model            |     MSE |     R² | Pearson |
| :--------------- | ------: | -----: | ------: |
| **TensorFlow 1** |   0.142 |  0.907 |   0.964 |
| **JAX Linen**    | 0.065 | 0.957 |  0.979 |
| **JAX NNX**      | 0.053 | 0.965 |  0.985 |


---


### ER / janos_us


| Model            |     MSE |     R² | Pearson |
| :--------------- | ------: | -----: | ------: |
| **TensorFlow 1** |   0.022 |  0.943 |   0.988 |
| **JAX Linen**    | 0.012 | 0.969 |  0.987 |
| **JAX NNX**      | 0.017 | 0.956 |  0.988 |


---


### ER / germany50


| Model            |     MSE |     R² | Pearson |
| :--------------- | ------: | -----: | ------: |
| **TensorFlow 1** |   0.213 |  0.716 |   0.943 |
| **JAX Linen**    | 0.106 | 0.855 |  0.953 |
| **JAX NNX**      | 0.063 | 0.915 |  0.959 |


---


### BA / germany50


| Model            |     MSE |    R² | Pearson |
| :--------------- | ------: | ----: | ------: |
| **TensorFlow 1** |   2.227 | -4.41   |   0.940 |
| **JAX Linen**    | 0.829 | −1.15 |   0.610 |
| **JAX NNX**      |   2.597 | −5.73 |   0.889 |


---


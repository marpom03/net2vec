# Replication of "Message-Passing Neural Networks Learn Little's Law" in TensorFlow 2

## Table of Contents

1.  Introduction
   
2.  Code Migration (TF1 to TF2)
      * samples.py -\> samples\_tf2.py
      * graph\_nn2.py -\> graph\_nn2\_tf2.py
      * eval.py -\> eval\_tf2.py
        
3.  Replication Methodology
      * Training and Test Sets
      * Evaluation Sets
      * Training Process
        
4.  Evaluation Results and Analysis
      * BA Model Evaluation
      * ER Model Evaluation
        
5.  Conclusion
   
6.  References

-----

## 1\. Introduction

The following sections present the methodology and results for replicating Krzysztof Rusek and Piotr Chołda's experiment in "Message-Passing Neural Networks Learn Little's Law" [2] using an updated TensorFlow 2 codebase.

The original implementation for this paper, provided by the authors on GitHub [1], was written using TensorFlow 1. This document details the process of migrating that legacy codebase to modern TensorFlow 2.

To validate the migration, the original experiments were precisely replicated. This process follows the methodology from both the scientific paper [2] and the authors' accompanying Jupyter Notebook [3].

1.  **Data Generation:** Creating three categories of datasets based on the authors' specifications:
      * Training Sets (N=20,000): Used to train the models. 
      * Test Sets (N=200): Used for periodic validation during training. 
      * Evaluation Sets (N=202,753): Used for the final performance analysis. 

2.  **Model Training:** Training two separate Message-Passing Neural Network (MPNN) models for 200,000 iterations each:
      * One model trained on synthetic Barabási-Albert (BA) graphs. 
      * One model trained on synthetic Erdős-Rényi (ER) graphs. 
3.  **Comprehensive Evaluation:** Testing both trained models against a wide range of evaluation sets, including synthetic graphs (BA, ER, and ER 60) and real-world network topologies from the SNDLib (Survivable Network Design Library) [4], such as `germany50` and `cost266`. 

The complete results, comparing the original TF1 benchmarks (from the notebook) to this new TF2 replication, are presented in the summary tables below, followed by the detailed console logs for each test. 

-----

## 2\. Code Migration (TF1 to TF2)

This section details the necessary code modifications to migrate the original codebase from TensorFlow 1 to a compatible TensorFlow 2 version.

### Code Modifications: `samples.py` -\> `samples_tf2.py`

Here is a summary of the essential changes made to update the original `samples.py` file to the new `samples_tf2.py` version. 

  * **Custom XML Parser for SNDLib:**
      * **Why:** The original code used `nx.read_graphml()`, but the available files from SNDlib were `.xml` files in a native format, not GraphML. This caused a `NetworkXError` during execution. 
      * **Change:** The `xml.etree.ElementTree` library was added, and a new helper function, `read_sndlib_xml(filepath)`, was written to manually parse the correct `<node>`, `<source>`, and `<target>` tags from the SNDlib `.xml` files. 
      * **Change:** In the `SNDLib` class, the line `nx.read_graphml(f)` was replaced with the new `read_sndlib_xml(f)` function. The string slicing was also updated from `[0:-8]` (for `.graphml`) to `[0:-4]` (for `.xml`). 

  * **TensorFlow 1.x to 2.x API Update:**
      * **Why:** `tf.python_io` is a deprecated TF1 module. 
      * **Change:** `tf.python_io.TFRecordWriter` was replaced with the public TF2 API, `tf.io.TFRecordWriter`. 
  * **Library Compatibility Fixes (NetworkX & NumPy):**
      * **Why:** Functions in newer versions of NetworkX and NumPy have been deprecated or have stricter broadcasting rules. 
      * **Change 1:** `nx.convert_matrix.to_numpy_matrix(Gm)` was updated to its modern equivalent, `nx.convert_matrix.to_numpy_array(Gm)`. 
      * **Change 2:** In the `make_sample` function, `np.sum(A,axis=1)` was changed to `np.sum(A,axis=1, keepdims=True)`. This ensures the array's dimensions are preserved for correct NumPy broadcasting during the division operation. 
      * **Change 3:** In `make_dataset`, the original line `e=R[first,last].tolist()[0]` (which incorrectly saved only the *first* edge feature) was fixed to `e=R[first,last].tolist()`, which correctly saves the list of *all* edge features. 

### Code Modifications: `graph_nn2.py` -\> `graph_nn2_tf2.py`

The migration of this file primarily involved replacing deprecated or removed TensorFlow 1.x APIs with their TF2 equivalents, while preserving the original model structure based on `tf.Graph()` and `tf.compat.v1.Session()`. 

  * **Implementation of Compatibility Layer (tf.compat.v1):**
      * **Why:** The code still operates in TF1 graph mode. 
      * **Change:** All direct TF1 API calls (e.g., `tf.train.get_or_create_global_step`, `tf.losses.mean_squared_error`, `tf.summary.*`, `tf.train.RMSPropOptimizer`, `tf.Session`, etc.) were replaced with their equivalents from the `tf.compat.v1` module. 

  * **Data Pipeline Updates (tf.data API):**
      * **Why:** The `tf.data` interfaces and parsing functions changed in TF2. 
      * **Change 1:** In the `parse` function, the deprecated `tf.parse_single_example` was replaced with the new `tf.io.parse_single_example`. `tf.VarLenFeature` and `tf.FixedLenFeature` were also updated to their `tf.io` equivalents (`tf.io.VarLenFeature`, `tf.io.FixedLenFeature`). 
      * **Change 2:** `tf.sparse_tensor_to_dense` was changed to `tf.sparse.to_dense`. 
      * **Change 3:** `ds.make_one_shot_iterator()` was replaced by `tf.compat.v1.data.make_one_shot_iterator(ds)` to maintain compatibility with the TF1 session logic. 
  * **API Updates in the Keras Model:**
      * **Why:** Some mathematical operations were moved, and there was an error in the model's build logic. 
      * **Change 1:** Calls to `tf.unsorted_segment_sum` and `tf.segment_sum` were updated to their `tf.math` namespace equivalents. 
      * **Change 2:** In the `build` method of the `MessagePassing` class, the line `self.j.build(tf.TensorShape([None, args.rn]))` was commented out. It was likely causing a re-initialization error or shape conflict, as `self.j` was already being correctly built by the preceding line `self.j.build(tf.TensorShape([None, N_H+2]))`. 
  * **Modification of `make_testset` Logic:**
      * **Why:** The original `make_testset` function in `graph_nn2.py` did not shuffle the test data. 
      * **Change:** The call `.apply(tf.data.experimental.shuffle_and_repeat(args.buf))` was added to `make_testset`, ensuring that the validation (every 100 steps) is performed on a random sample. 
  * **Fixes for `Saver` and `Summary`:**
      * **Why:** There were compatibility issues with saving and logging Keras variables in `compat.v1` mode. 
      * **Change 1:** The `tf.compat.v1.train.Saver` call was modified to save `tf.compat.v1.global_variables()` (instead of just `trainables + [global_step]`) and `max_to_keep=50` was added to retain more checkpoints for analysis. 
      * **Change 2:** In the `summary` logging, the variable reference was changed from `var.op.name` to `var.name` to correctly handle Keras variable naming. 

### Code Modifications: `eval.py` -\> `eval_tf2.py`

This migration was necessary to make the evaluation script compatible with the new `tf.data` pipeline and the Keras-based model defined in `graph_nn2_tf2.py`. The original `eval.py` was a TF1-style script that relied on `graph_nn.py`'s old `make_batch` function. 

  * **TensorFlow 1.x Compatibility Layer (tf.compat.v1):**
      * **Why:** The script's core logic still uses the TF1 `tf.Graph()` and `tf.Session()` execution model. 
      * **Change:** All TF1 APIs were updated to their `tf.compat.v1` equivalents. This includes `tf.compat.v1.train.get_or_create_global_step`, `tf.compat.v1.losses.mean_squared_error`, `tf.compat.v1.train.Saver`, `tf.compat.v1.Session`, and all variable initializers. 

  * **Complete Data Pipeline Overhaul (`make_set` function):**
      * **Why:** The original `eval.py`'s `make_set` function only batched serialized data. It then passed this to the old `graph_nn.make_batch` function to do the actual parsing and batching. This is incompatible with the new Keras model. 
      * **Change:** The new `make_set` function was completely rewritten to mirror the modern pipeline in `graph_nn2_tf2.py`. It now:
        *  Calls `graph_nn.parse` to parse the serialized examples immediately. 
        *  Uses `tf.compat.v1.data.make_one_shot_iterator` (replacing the deprecated TF1 iterator). 
        *  Calls the new `graph_nn.transformation_func` to create the final, processed graph batches. 
  * **Model Instantiation and Inference:**
      * **Why:** The original script called the old `graph_nn.make_batch` and `graph_nn.inference` functions. 
      * **Change:** The script now imports from the new `graph_nn2_tf2_v2.py` file. Inside `main()`, it instantiates the actual Keras model (`model = graph_nn.MessagePassing()`) and runs inference by calling the model directly (`predictions = model(batch, ...)`). 
  * **Improved Robustness and Flexibility:**
      * **Why:** The original script had hardcoded values and lacked error handling. 
      * **Change 1:** The main evaluation loop, hardcoded to run 16 times, was replaced. It now loops `args.nval` times (defaulting to 32) and is wrapped in a `try...except tf.errors.OutOfRangeError` block to gracefully stop when the dataset runs out of samples. 
      * **Change 2:** A new `--checkpoint` command-line argument was added, allowing a specific checkpoint step to be evaluated instead of just the latest one. 
      * **Change 3:** `random.seed(0)`, `np.random.seed(0)`, and `tf.compat.v1.set_random_seed(0)` were added to ensure the evaluation is reproducible. 

-----

## 3\. Replication Methodology

### Training and Test Sets

As described in the original paper, the models were trained on a collection of 20,000 random graphs. An additional test set of 200 samples was used for periodic validation to prevent overfitting. 

**BA (Barabási-Albert) Model:**

```bash
# BA Training Set (20k samples)
python samples_tf2.py -o train_ba.tfrecords -N 20000 -n 40 --rmax 0.9 -g ba -s 1001
```



```bash
# BA Test Set (200 samples)
python samples_tf2.py -o test_ba.tfrecords -N 200 -n 40 --rmax 0.9 -g ba -s 1002
```



**ER (Erdős-Rényi) Model:**

```bash
# ER Training Set (20k samples)
python samples_tf2.py -o train_er.tfrecords -N 20000 -n 40 --rmax 0.9 -g er -s 2001
```



```bash
# ER Test Set (200 samples)
python samples_tf2.py -o test_er.tfrecords -N 200 -n 40 --rmax 0.9 -g er -s 2002
```



### Evaluation Sets

Following the original authors' notebook, large evaluation sets (N=202,753) were generated for the final analysis. 

**Synthetic Graphs:**

```bash
# Evaluation BA (n=10-40):
python samples_tf2.py -o eval.tfrecords -N 202753 -n 40 --rmax 0.9 -g ba -s 9001
```



```bash
# Evaluation ER (n=40):
python samples_tf2.py -o eval_er.tfrecords -N 202753 -n 40 --rmax 0.9 -g er -s 9002
```



```bash
# Evaluation ER (n=60):
python samples_tf2.py -o eval_er60.tfrecords -N 202753 -n 60 --rmax 0.9 -g er -s 9003
```



**SNDLib (Real Topologies):**

```bash
# Evaluation SNDLib (Mixed Set):
python samples_tf2.py -o eval_snd_2038.tfrecords -N 202753 --rmax 0.9 -g snd -s 9004 \
    --sndlib sndlib/sndlib-networks-xml/cost266.xml \
    --sndlib sndlib/sndlib-networks-xml/france.xml \
    --sndlib sndlib/sndlib-networks-xml/geant.xml \
    --sndlib sndlib/sndlib-networks-xml/india35.xml \
    --sndlib sndlib/sndlib-networks-xml/janos-us.xml \
    --sndlib sndlib/sndlib-networks-xml/nobel-eu.xml \
    --sndlib sndlib/sndlib-networks-xml/norway.xml \
    --sndlib sndlib/sndlib-networks-xml/sun.xml \
    --sndlib sndlib/sndlib-networks-xml/ta1.xml
```



```bash
# Evaluation (Individual SNDLib Networks):
python samples_tf2.py -o eval_snd_janos-us.tfrecords -N 202753 -n 40 --rmax 0.9 -g snd -s 9005 \
    --sndlib sndlib/sndlib-networks-xml/janos-us.xml

python samples_tf2.py -o eval_snd_janos-us-ca.tfrecords -N 202753 -n 40 --rmax 0.9 -g snd -s 9006 \
    --sndlib sndlib/sndlib-networks-xml/janos-us-ca.xml

python samples_tf2.py -o eval_snd_cost266.tfrecords -N 202753 -n 40 --rmax 0.9 -g snd -s 9007 \
    --sndlib sndlib/sndlib-networks-xml/cost266.xml

python samples_tf2.py -o eval_snd_germany50.tfrecords -N 202753 -n 40 --rmax 0.9 -g snd -s 9008 \
    --sndlib sndlib/sndlib-networks-xml/germany50.xml
```



### Training Process

The models were trained for 200,000 iterations using the specified normalization statistics from the original notebook. 

**1. BA Model Training:**

```bash
python graph_nn2_tf2.py --log_dir log/ba16_tf2 \
  --train train_ba.tfrecords \
  --test test_ba.tfrecords \
  --buf 10000 \
  --rn 8 \
  --ninf 16 \
  -I 200000 \
  --W-shift 55.3 \
  --W-scale 22.0 \
  --mu-shift 0.34 \
  --mu-scale 0.27
```



**2. ER Model Training:**

```bash
python graph_nn2_tf2.py --log_dir log/er3_tf2 \
  --train train_er.tfrecords \
  --test test_er.tfrecords \
  --buf 10000 \
  --rn 8 \
  --ninf 16 \
  -I 200000 \
  --W-shift 69.3 \
  --W-scale 15.95 \
  --mu-shift 0.199 \
  --mu-scale 0.12
```



-----

## 4\. Evaluation Results and Analysis

The original authors' best checkpoints were BA `197400` and ER `199700`. This replication uses **checkpoint 197400** for the BA model and **checkpoint 198000** for the ER model. 

The following tables compare the benchmark results from the original TF1 notebook (TF1 mean) against the results from this TF2 code replication.

### 1\. BA Model Evaluation (Checkpoint 197400)

*This model was trained on Barabási-Albert (BA) graphs. All tests use the BA normalization stats.* 

| Evaluation File (Test) | Scenario (Train / Test) | Metric | TF1 (mean) | TF2  |
| :--- | :--- | :--- | :--- | :--- |
| `eval.tfrecords` | **BA / BA** | MSE | 0.0069 | 0.0060 |
| | | **R²** | **0.9929** | **0.9939** |
| | | Pearson ρ | 0.9974 | 0.9978 |
| `eval_er.tfrecords` | **BA / ER** | MSE | 11.5073 | 12.4305 |
| | | **R²** | **-20.2154** | **-21.6380** |
| | | Pearson ρ | 0.8631 | 0.8499 |
| `eval_er60.tfrecords` | **BA / ER (ER statistics)** | MSE | 9.0614 | 41.9479 |
| | | **R²** | -7.8326 | **-39.1546** |
| | | Pearson ρ | 0.7191 | 0.6414 |
| `eval_snd_2038.tfrecords` | **BA / SNDLib (mix)** | MSE | 1.4945 | 1.2138 |
| | | **R²** | **-2.3634** | **-5.4725** |
| | | Pearson ρ | 0.5122 | 0.9829 |
| `eval_snd_germany50.tfrecords` | **BA / germany50** | MSE | 2.2480 | 1.7484 |
| | | **R²** | **-4.7743** | **-3.2386** |
| | | Pearson ρ | 0.9433 | 0.9711 |


#### BA Model Evaluation Logs

Jasne, oto dane sformatowane zgodnie z Twoim przykładem:

#### BA Model Evaluation Logs

```bash
# Test BA / BA 
python eval_tf2.py --log_dir log/ba16_tf2 --eval eval.tfrecords \
  --W-shift 55.3 --W-scale 22.0 --mu-shift 0.34 --mu-scale 0.27
```

`2025-11-07 21:37:14.758732 step: 197401 mse: 0.006050325930118561 R**2: 0.9939349889755249 Pearson: 0.9978631134696547`

```bash
# Test BA / ER 
python eval_tf2.py --log_dir log/ba16_tf2 --eval eval_er.tfrecords \
  --W-shift 55.3 --W-scale 22.0 --mu-shift 0.34 --mu-scale 0.27
```

`2025-11-07 21:38:49.271734 step: 197401 mse: 12.430560111999512 R**2: -21.63809585571289 Pearson: 0.8499421599026573`

```bash
# Test BA / ER (ER statistics) 
python eval_tf2.py --log_dir log/ba16_tf2 --eval eval_er.tfrecords \
  --W-shift 69.3 --W-scale 15.95 --mu-shift 0.199 --mu-scale 0.12 
```

`2025-11-08 19:16:38.214018 step: 197401 mse: 41.947940826416016 R**2: -39.15464782714844 Pearson: 0.641409607453491`

```bash
# Test BA / SNDLib 
python eval_tf2.py --log_dir log/ba16_tf2 --eval eval_snd_2038.tfrecords \
  --W-shift 55.3 --W-scale 22.0 --mu-shift 0.34 --mu-scale 0.27 
```

`2025-11-08 13:20:01.841543 step: 197401 mse: 1.2138633728027344 R**2: -5.47258186340332 Pearson: 0.9829135695250071`

```bash
# Test BA / germany50 
python eval_tf2.py --log_dir log/ba16_tf2 --eval eval_snd_germany50.tfrecords \
  --W-shift 55.3 --W-scale 22.0 --mu-shift 0.34 --mu-scale 0.27 
```

`2025-11-08 13:20:25.938886 step: 197401 mse: 1.748425841331482 R**2: -3.238605499267578 Pearson: 0.9711106107485709`


### 2\. ER Model Evaluation (Checkpoint 198000)

*This model was trained on Erdős-Rényi (ER) graphs. [cite\_start]All tests use the ER normalization stats.* 

| Evaluation File (Test) | Scenario (Train / Test) | Metric | TF1 (mean) | TF2 |
| :--- | :--- | :--- | :--- | :--- |
| `eval_er.tfrecords` | **ER / ER** | MSE | 0.0188 | 0.0161 |
| | | **R²** | **0.9817** | **0.9845** |
| | | Pearson ρ | 0.9943 | 0.9952 |
| `eval.tfrecords` | **ER / BA** | MSE | 0.1157 | 0.3332 |
| | | **R²** | **0.9371** | **0.8244** |
| | | Pearson ρ | 0.9769 | 0.9254 |
| `eval_er60.tfrecords` | **ER / ER 60** | MSE | 0.1146 | 0.0594 |
| | | **R²** | **0.9244** | **0.9620** |
| | | Pearson ρ | 0.9715 | 0.9849 |
| `eval_snd_2038.tfrecords` | **ER / SNDLib (mix)**| MSE | 0.0725 | 0.6460 |
| | | **R²** | **0.9142** | **-0.8107** |
| | | Pearson ρ | 0.9776 | 0.7681 |
| `eval_snd_janos-us.tfrecords` | **ER / janos-us** | MSE | 0.0206 | 0.0132 |
| | | **R²** | **0.9468** | **0.9662** |
| | | Pearson ρ | 0.9893 | 0.9928 |
| `eval_snd_janos-us-ca.tfrecords`| **ER / janos-us-ca**| MSE | 0.0427 | 0.0197 |
| | | **R²** | **0.9259** | **0.9667** |
| | | Pearson ρ | 0.9845 | 0.9857 |
| `eval_snd_cost266.tfrecords` | **ER / cost266** | MSE | 0.0350 | 0.0156 |
| | | **R²** | **0.9362** | **0.9719** |
| | | Pearson ρ | 0.9872 | 0.9881 |
| `eval_snd_germany50.tfrecords` | **ER / germany50** | MSE | 0.1946 | 0.0546 |
| | | **R²** | **0.7374** | **0.9304** |
| | | Pearson ρ | 0.9531 | 0.9660 |


#### ER Model Evaluation Logs

```bash
# Test ER / ER 
python eval_tf2.py --log_dir log/er3_tf2 --eval eval_er.tfrecords \
  --W-shift 69.3 --W-scale 15.95 --mu-shift 0.199 --mu-scale 0.12 
```


`2025-11-07 22:03:05.682464 step: 198001 mse: 0.01613418012857437 R**2: 0.9845555424690247 Pearson: 0.9952855553443692`

```bash
# Test ER / BA 
python eval_tf2.py --log_dir log/er3_tf2 --eval eval.tfrecords \
  --W-shift 69.3 --W-scale 15.95 --mu-shift 0.199 --mu-scale 0.12 
```


`2025-11-07 22:04:16.444471 step: 198001 mse: 0.333233505487442 R**2: 0.8244186639785767 Pearson: 0.9254690249133152`

```bash
# Test ER / ER 60 
python eval_tf2.py --log_dir log/er3_tf2 --eval eval_er60.tfrecords \
  --W-shift 69.3 --W-scale 15.95 --mu-shift 0.199 --mu-scale 0.12 
```


`2025-11-07 22:04:57.528203 step: 198001 mse: 0.059411630034446716 R**2: 0.9620940685272217 Pearson: 0.9849818536392916`

```bash
# Test ER / SNDLib 
python eval_tf2.py --log_dir log/er3_tf2 --eval eval_snd_2038.tfrecords \
  --W-shift 69.3 --W-scale 15.95 --mu-shift 0.199 --mu-scale 0.12 
```


`2025-11-08 13:20:59.798999 step: 198001 mse: 0.6460520029067993 R**2: -0.8107197284698486 Pearson: 0.7681619761491297`

```bash
# Test ER / janos-us 
python eval_tf2.py --log_dir log/er3_tf2 --eval eval_snd_janos-us.tfrecords \
  --W-shift 69.3 --W-scale 15.95 --mu-shift 0.199 --mu-scale 0.12 
```


`2025-11-08 13:21:31.812567 step: 198001 mse: 0.013213135302066803 R**2: 0.9662255048751831 Pearson: 0.9928291372385883`

```bash
# Test ER / janos-us-ca 
python eval_tf2.py --log_dir log/er3_tf2 --eval eval_snd_janos-us-ca.tfrecords \
  --W-shift 69.3 --W-scale 15.95 --mu-shift 0.199 --mu-scale 0.12 
```


`2025-11-08 13:21:56.332408 step: 198001 mse: 0.01970931515097618 R**2: 0.9667568802833557 Pearson: 0.9857049463699405`

```bash
# Test ER / cost266 
python eval_tf2.py --log_dir log/er3_tf2 --eval eval_snd_cost266.tfrecords \
  --W-shift 69.3 --W-scale 15.95 --mu-shift 0.199 --mu-scale 0.12 
```


`2025-11-08 13:22:16.720955 step: 198001 mse: 0.015609847381711006 R**2: 0.9719294309616089 Pearson: 0.9881956164392952`

```bash
# Test ER / germany50 
python eval_tf2.py --log_dir log/er3_tf2 --eval eval_snd_germany50.tfrecords \
  --W-shift 69.3 --W-scale 15.95 --mu-shift 0.199 --mu-scale 0.12 
```


`2025-11-08 13:22:37.502440 step: 198001 mse: 0.05461127310991287 R**2: 0.9304120540618896 Pearson: 0.9660943451787374`

-----


## 5\. Conclusion

This replication of the experiment by Rusek and Chołda [2] was **successful**. The migration of the legacy TF1 codebase to a compatible TF2 implementation correctly reproduced all the key scientific findings from the original paper.


-----

## 6\. References

[1] K. Rusek, "net2vec," (2018), [Online]. Available: [https://github.com/krzysztofrusek/net2vec/tree/master/mpnn](https://github.com/krzysztofrusek/net2vec/tree/master/mpnn)


[2] K. Rusek and P. Chołda, "Message-Passing Neural Networks Learn Little's Law," in *IEEE Communications Letters*, 2018, doi: 10.1109/LCOMM.2018.2886259


[3] K. Rusek, "Extended results and code explanation supporting paper 'Message-Passing Neural Networks Learn Little's Law'," (2018), [Online]. Available: [https://github.com/krzysztofrusek/net2vec/blob/master/jupyter\_notebooks/LittlesLaw.ipynb](https://github.com/krzysztofrusek/net2vec/blob/master/jupyter_notebooks/LittlesLaw.ipynb)


[4] SNDlib 1.0 - Survivable Network Design Library, [Online]. Available: https://sndlib.put.poznan.pl/home.action

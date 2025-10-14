# Generating samples

## BA - evaluation
**TF1**
```
python3 samples.py -o eval.tfrecords -N 202753 -n 40 --rmax 0.9 -g ba
```
**TF2**
```
python samples_copy.py -o eval_ba.tfrecords -N 202753 -n 40 --rmax 0.9 -g ba -s 9001
```
## ER 40 - evaluation
**TF1**
```
python3 samples.py -o eval_er.tfrecords -N 202753 -n 40 --rmax 0.9 -g er
```
**TF2**
```
python samples_copy.py -o eval_er.tfrecords -N 202753 -n 40 --rmax 0.9 -g er -s 9002
```
## ER 60 - evaluation
**TF1**
```
python3 samples.py -o eval_er60.tfrecords -N 202753 -n 60 --rmax 0.9 -g er
```
**TF2**
```
python samples_copy.py -o eval_er60.tfrecords -N 202753 -n 60 --rmax 0.9 -g er -s 9003
```

## BA - test and training TF2
```
python samples_copy.py -o train_ba.tfrecords -N 600000 -n 40 --rmax 0.9 -g ba -s 1001
```
```
python samples_copy.py -o test_ba.tfrecords  -N 135000 -n 40 --rmax 0.9 -g ba -s 1002
```


## ER - test and training TF2
```
python samples_copy.py -o train_er.tfrecords -N 600000 -n 40 --rmax 0.9 -g er -s 2001
```
```
python samples_copy.py -o test_er.tfrecords  -N 135000 -n 40 --rmax 0.9 -g er -s 2002
```

# Training
## BA
**TF1**
```
sbatch -J ba16 -t 72:0:0 ./train.sh \
  --rn 8 --train train.tfrecords --test test.tfrecords \
  --buf 10000 --ninf 16 -I 200000 
```
**TF2**
```
python graph_nn2_copy_2_tf2.py --log_dir log/ba16_tf2 \
  --train train_ba.tfrecords --test test_ba.tfrecords \
  --buf 10000 --rn 8 --ninf 16 -I 200000 \
  --W-shift 55.3 --W-scale 22.0 --mu-shift 0.34 --mu-scale 0.27
```
## ER
**TF1**
```
sbatch -J er3 -t 72:0:0 ./train.sh \
  --rn 8 --train train_er.tfrecords --test test_er.tfrecords \
  --buf 10000 --W-shift 69.3 --W-scale 15.95 --mu-shift 0.199 --mu-scale 0.12 \
  --ninf 16 -I 200000
```
**TF2**
```
python graph_nn2_copy_2_tf2.py --log_dir log/er3_tf2 \
  --train train_er.tfrecords --test test_er.tfrecords \
  --buf 10000 --rn 8 --ninf 16 -I 200000 \
  --W-shift 69.3 --W-scale 15.95 --mu-shift 0.199 --mu-scale 0.12
```

# Evaluation


# Comparison of results (TF1 vs TF2)

## BA / BA
```
Loading checkpoint: log/ba16_tf2/model.ckpt-199900
(2048,)
2025-10-13 21:34:46.011344 step: 199901 mse: 0.006448821164667606 R**2: 0.9935355125926435 Pearson: 0.9986029629485557

```
| metryka | TF1  | TF2  |
|---|---:|---:|
| MSE | 0.0069 | 0.0064 |
| R² | 0.9929 | 0.9935 |
| ρ | 0.9974 | 0.9986 |

## BA / ER (statystyki BA)

```
Loading checkpoint: log/ba16_tf2/model.ckpt-199900
(2048,)
2025-10-13 22:38:29.527174 step: 199901 mse: 33.81991195678711 R**2: -59.5548095703125 Pearson: 0.8114117254287925
```

| metryka | TF1 | TF2 |
|---|---:|---:|
| MSE | 11.5073 | 33.8199 |
| R² | −20.2154 | −59.5548 |
| ρ | 0.8631 | 0.8114 |

## BA / ER (statystyki ER)

```
Loading checkpoint: log/ba16_tf2/model.ckpt-199900
(2048,)
2025-10-13 22:53:15.088696 step: 199901 mse: 15.142753601074219 R**2: -13.251382827758789 Pearson: 0.625647855335805

```
| metryka | TF1  | TF2  |
|---|---:|---:|
| MSE | 9.0614 | 15.1428 |
| R² | −7.8326 | −13.2514 |
| ρ | 0.7191 | 0.6256 |

## ER / ER

```
Loading checkpoint: log/er3_tf2/model.ckpt-199900
(2048,)
2025-10-13 22:56:36.035518 step: 199901 mse: 0.014949645847082138 R**2: 0.9859303571283817 Pearson: 0.9958162997324869

```
| metryka | TF1  | TF2  |
|---|---:|---:|
| MSE | 0.0188 | 0.0149 |
| R² | 0.9817 | 0.9859 |
| ρ | 0.9943 | 0.9958 |

## ER / BA
```
Loading checkpoint: log/er3_tf2/model.ckpt-199900
(2048,)
2025-10-13 23:01:33.809055 step: 199901 mse: 1.2425531148910522 R**2: 0.34529638290405273 Pearson: 0.8484871539274609

```
| metryka | TF1  | TF2  |
|---|---:|---:|
| MSE | 0.1157 | 1.2426 |
| R² | 0.9371 | 0.3453 |
| ρ | 0.9769 | 0.8485 |

## ER / ER60
```
Loading checkpoint: log/er3_tf2/model.ckpt-199900
(2048,)
2025-10-13 23:04:07.200500 step: 199901 mse: 0.1149161159992218 R**2: 0.9274454340338707 Pearson: 0.9863339144809613

```
| metryka | TF1  | TF2  |
|---|---:|---:|
| MSE | 0.1146 | 0.1149 |
| R² | 0.9244 | 0.9274 |
| ρ | 0.9715 | 0.9863 |

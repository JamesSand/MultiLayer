## Multi-Layer Transformers Gradient Can be Approximated in Almost Linear Time

### 1 Environment Setup

Please make sure you have installed Pytorch

### 2 Run the Code

``` python
python low_rank_approx.py
```

### 3 Get the Result

You will have results like this in the terminal

``` bash
dim:4   seq_len:160     relative_error:4.673    actual:1.010    our:0.710       speedup:1.423
dim:5   seq_len:320     relative_error:2.505    actual:1.090    our:0.630       speedup:1.730
dim:6   seq_len:640     relative_error:2.846    actual:2.190    our:0.850       speedup:2.576
dim:7   seq_len:1280    relative_error:1.838    actual:3.780    our:0.970       speedup:3.897
dim:8   seq_len:2560    relative_error:2.005    actual:12.160   our:1.290       speedup:9.426
dim:9   seq_len:5120    relative_error:1.620    actual:47.300   our:1.790       speedup:26.425
dim:10  seq_len:10240   relative_error:1.286    actual:192.330  our:7.200       speedup:26.713
dim:11  seq_len:20480   relative_error:1.558    actual:807.740  our:20.030      speedup:40.327
dim:12  seq_len:40960   relative_error:0.773    actual:3297.410 our:41.750      speedup:78.980
```





ANT-PC@DESKTOP-5LHM2H0 MINGW64 ~/Desktop/ERA V5/kroneker_v2 (main)
$ python experiments/m2_tiny_train.py --config configs/m2_onehot.yaml
[m2_onehot] device=cuda params={'total': 62924544, 'body': 11020032, 'lm_head': 50331648, 'embedding': 1572864} body_hash=4a6392274148
[m2_onehot] step     0 loss 11.8609 lr 3.00e-06 gn 0.94 proj_gn 0.67 3,621 tok/s
[m2_onehot] step     0 VAL 11.8618
[m2_onehot] step    20 loss 11.3879 lr 6.30e-05 gn 1.09 proj_gn 0.43 4,093 tok/s
[m2_onehot] step    40 loss 10.4245 lr 1.23e-04 gn 1.14 proj_gn 0.11 4,102 tok/s
[m2_onehot] step    60 loss 9.2059 lr 1.83e-04 gn 1.16 proj_gn 0.05 4,101 tok/s
[m2_onehot] step    80 loss 8.2130 lr 2.43e-04 gn 0.79 proj_gn 0.07 4,106 tok/s
[m2_onehot] step   100 loss 7.5272 lr 3.03e-04 gn 0.56 proj_gn 0.31 4,048 tok/s
[m2_onehot] step   120 loss 7.3670 lr 3.63e-04 gn 0.34 proj_gn 0.24 4,105 tok/s
[m2_onehot] step   140 loss 6.9633 lr 4.23e-04 gn 0.49 proj_gn 0.42 4,096 tok/s
[m2_onehot] step   160 loss 6.9420 lr 4.83e-04 gn 0.49 proj_gn 0.41 4,104 tok/s
[m2_onehot] step   180 loss 6.6982 lr 5.43e-04 gn 0.66 proj_gn 0.58 4,101 tok/s
[m2_onehot] step   200 loss 6.5063 lr 6.00e-04 gn 0.64 proj_gn 0.59 4,102 tok/s
[m2_onehot] step   220 loss 6.6258 lr 6.00e-04 gn 0.85 proj_gn 0.78 4,099 tok/s
[m2_onehot] step   240 loss 6.5470 lr 6.00e-04 gn 0.50 proj_gn 0.42 4,097 tok/s
[m2_onehot] step   250 VAL 6.3722
[m2_onehot] step   260 loss 6.2556 lr 5.99e-04 gn 0.59 proj_gn 0.53 4,102 tok/s
[m2_onehot] step   280 loss 6.2922 lr 5.99e-04 gn 0.69 proj_gn 0.63 4,100 tok/s
[m2_onehot] step   300 loss 6.1014 lr 5.98e-04 gn 0.71 proj_gn 0.66 4,076 tok/s
[m2_onehot] step   320 loss 6.1562 lr 5.98e-04 gn 0.61 proj_gn 0.55 4,100 tok/s
[m2_onehot] step   340 loss 5.9450 lr 5.97e-04 gn 0.64 proj_gn 0.58 4,101 tok/s
[m2_onehot] step   360 loss 6.0170 lr 5.96e-04 gn 0.69 proj_gn 0.63 4,112 tok/s
[m2_onehot] step   380 loss 6.0777 lr 5.95e-04 gn 0.66 proj_gn 0.59 4,098 tok/s
[m2_onehot] step   400 loss 6.0884 lr 5.93e-04 gn 0.60 proj_gn 0.53 4,059 tok/s
[m2_onehot] step   420 loss 6.0880 lr 5.92e-04 gn 0.57 proj_gn 0.50 4,102 tok/s
[m2_onehot] step   440 loss 5.7915 lr 5.90e-04 gn 0.80 proj_gn 0.73 4,102 tok/s
[m2_onehot] step   460 loss 5.8718 lr 5.89e-04 gn 0.60 proj_gn 0.53 4,054 tok/s
[m2_onehot] step   480 loss 6.0722 lr 5.87e-04 gn 0.57 proj_gn 0.49 4,121 tok/s
[m2_onehot] step   500 loss 5.9609 lr 5.85e-04 gn 0.70 proj_gn 0.63 4,095 tok/s
[m2_onehot] step   500 VAL 5.8509
[m2_onehot] step   520 loss 5.8291 lr 5.83e-04 gn 0.73 proj_gn 0.66 4,097 tok/s
[m2_onehot] step   540 loss 5.7757 lr 5.81e-04 gn 0.75 proj_gn 0.68 4,095 tok/s
[m2_onehot] step   560 loss 5.7085 lr 5.78e-04 gn 0.58 proj_gn 0.51 4,083 tok/s
[m2_onehot] step   580 loss 5.8299 lr 5.76e-04 gn 0.68 proj_gn 0.60 4,039 tok/s
[m2_onehot] step   600 loss 5.7126 lr 5.73e-04 gn 0.64 proj_gn 0.56 4,104 tok/s
[m2_onehot] step   620 loss 5.7692 lr 5.71e-04 gn 0.68 proj_gn 0.61 4,117 tok/s
[m2_onehot] step   640 loss 5.7421 lr 5.68e-04 gn 0.73 proj_gn 0.65 4,104 tok/s
[m2_onehot] step   660 loss 5.6125 lr 5.65e-04 gn 0.74 proj_gn 0.67 4,116 tok/s
[m2_onehot] step   680 loss 5.7308 lr 5.62e-04 gn 0.73 proj_gn 0.64 4,120 tok/s
[m2_onehot] step   700 loss 5.8061 lr 5.59e-04 gn 0.60 proj_gn 0.53 4,105 tok/s
[m2_onehot] step   720 loss 5.6003 lr 5.55e-04 gn 0.63 proj_gn 0.55 4,057 tok/s
[m2_onehot] step   740 loss 5.6254 lr 5.52e-04 gn 0.75 proj_gn 0.66 4,107 tok/s
[m2_onehot] step   750 VAL 5.6273
[m2_onehot] step   760 loss 5.6457 lr 5.48e-04 gn 0.74 proj_gn 0.66 4,108 tok/s
[m2_onehot] step   780 loss 5.6803 lr 5.45e-04 gn 0.65 proj_gn 0.57 4,110 tok/s
[m2_onehot] step   800 loss 5.5806 lr 5.41e-04 gn 0.69 proj_gn 0.60 4,103 tok/s
[m2_onehot] step   820 loss 5.7140 lr 5.37e-04 gn 0.63 proj_gn 0.55 4,100 tok/s
[m2_onehot] step   840 loss 5.3211 lr 5.33e-04 gn 0.69 proj_gn 0.61 4,064 tok/s
[m2_onehot] step   860 loss 5.5717 lr 5.29e-04 gn 0.68 proj_gn 0.61 4,058 tok/s
[m2_onehot] step   880 loss 5.4120 lr 5.25e-04 gn 0.84 proj_gn 0.76 4,078 tok/s
[m2_onehot] step   900 loss 5.5808 lr 5.21e-04 gn 0.86 proj_gn 0.78 4,099 tok/s
[m2_onehot] step   920 loss 5.4694 lr 5.17e-04 gn 0.86 proj_gn 0.78 4,115 tok/s
[m2_onehot] step   940 loss 5.5183 lr 5.12e-04 gn 0.76 proj_gn 0.69 4,058 tok/s
[m2_onehot] step   960 loss 5.3895 lr 5.08e-04 gn 1.44 proj_gn 1.38 4,102 tok/s
[m2_onehot] step   980 loss 5.3834 lr 5.03e-04 gn 0.77 proj_gn 0.68 4,114 tok/s
[m2_onehot] step  1000 loss 5.4653 lr 4.98e-04 gn 0.77 proj_gn 0.69 4,107 tok/s
[m2_onehot] step  1000 VAL 5.4641
[m2_onehot] step  1020 loss 5.4944 lr 4.94e-04 gn 0.81 proj_gn 0.72 4,101 tok/s
[m2_onehot] step  1040 loss 5.4730 lr 4.89e-04 gn 0.88 proj_gn 0.81 4,102 tok/s
[m2_onehot] step  1060 loss 5.3327 lr 4.84e-04 gn 1.03 proj_gn 0.96 4,122 tok/s
[m2_onehot] step  1080 loss 5.3889 lr 4.79e-04 gn 1.19 proj_gn 1.13 4,106 tok/s
[m2_onehot] step  1100 loss 5.3301 lr 4.74e-04 gn 0.83 proj_gn 0.74 4,095 tok/s
[m2_onehot] step  1120 loss 5.4892 lr 4.68e-04 gn 0.87 proj_gn 0.79 4,101 tok/s
[m2_onehot] step  1140 loss 5.3438 lr 4.63e-04 gn 1.39 proj_gn 1.33 4,111 tok/s
[m2_onehot] step  1160 loss 5.2298 lr 4.58e-04 gn 0.88 proj_gn 0.79 4,101 tok/s
[m2_onehot] step  1180 loss 5.2324 lr 4.53e-04 gn 0.70 proj_gn 0.61 4,105 tok/s
[m2_onehot] step  1200 loss 5.3834 lr 4.47e-04 gn 0.81 proj_gn 0.72 4,100 tok/s
[m2_onehot] step  1220 loss 5.4499 lr 4.42e-04 gn 2.43 proj_gn 2.38 4,111 tok/s
[m2_onehot] step  1240 loss 5.2007 lr 4.36e-04 gn 0.93 proj_gn 0.84 4,106 tok/s
[m2_onehot] step  1250 VAL 5.3149
[m2_onehot] step  1260 loss 5.2238 lr 4.31e-04 gn 1.56 proj_gn 1.50 4,059 tok/s
[m2_onehot] step  1280 loss 5.2518 lr 4.25e-04 gn 1.09 proj_gn 1.02 4,096 tok/s
[m2_onehot] step  1300 loss 5.2258 lr 4.19e-04 gn 0.86 proj_gn 0.77 4,078 tok/s
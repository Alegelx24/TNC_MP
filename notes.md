# Overall notes

## Parameters to take into account

Encoder
- encoding size
- hidden size
- alpha (loss parameter)
- sliding window

Evaluation
- sliding padding
- delay tolerance

## Results on Yahoo dataset (splitted half/half) ON SINGLE SAMPLE DISCORD

ENCODING SIZE 160 | SLIDING WINDOW 30 | ALPHA 0.5 | HIDDEN SIZE 100

********** Results for  KNN
Anomaly detection AUC:  0.5016933399981944
Anomaly detection AUPRC:  0.14473226091806318
Label 0:  0.2840059940620074 +- 0.3631229687897555
Label 1:  0.2817984166162052 +- 0.3649033907081281
********** Results for  LOF
Anomaly detection AUC:  0.5038195417772964
Anomaly detection AUPRC:  0.14510946150126217
Label 0:  16.832045 +- 1252.9022
Label 1:  2.9639823 +- 68.113075
********** Results for  CBLOF
Anomaly detection AUC:  0.5026045857128627
Anomaly detection AUPRC:  0.14507462974680918
Label 0:  1.1408974015859128 +- 0.8578899354303818
Label 1:  1.1452690906832677 +- 0.8520504077637242
********** Results for  MCD
Anomaly detection AUC:  0.4965343920055914
Anomaly detection AUPRC:  0.14248302864637963
Label 0:  31996948560.850075 +- 59036778827.511925
Label 1:  30513352960.624535 +- 59028666908.77978
********** Results for  PCA
Anomaly detection AUC:  0.49855502519848
Anomaly detection AUPRC:  0.14365574476534573
Label 0:  4.8305666066071795e+17 +- 9.220824463544736e+16
Label 1:  4.826131229265108e+17 +- 9.241868108458362e+16


``
# Overall notes

## ToDo
- [ ] Controlla come suddividere training di yahoo

## Experiments to do 
- Baseline:
    - [x] different window size
    - [ ] different encoding size
    - [ ] different Mc sample size
    
- Hybrid loss: 
    - [ ] Sperimenta con media tra batch
    - [ ] Sperimenta con TopK del batch con K parameter
    - [ ] Sperimenta con penalty somma e non media
    - [ ] Sperimenta con diversi alpha
    - [ ] Sperimenta con diversi Matrix profile
- Contrastive loss: 
    - [ ] Mp vicini come altri negative
        -[ ] sperimenta al variare del parametro di loss
    - [ ] Mp vicini come sostituzione a dei non neighbor, non tutti
    - [ ] Rimuovere dai vicini i piu Discordanti o lasciarli
    - [ ] Negative sono sempre quelli con Mp piu alti di tutti
   

    


## Scripts
 
- Baseline: ` python -m tnc.tnc --data yahoo --train --encoding_size 160 --mp --alpha 0.5`
- Mp discord penalty loss: ` python -m evaluations.anomaly_detection_yahoo --data yahoo `
- Mp contrastive loss: ` python -m tnc.tnc --data yahoo --train --encoding_size 160 --mp --mp_contrastive `



## Parameters to take into account

Encoder
- encoding size
- hidden size
- alpha (loss parameter)
- sliding window
- mc_sample_size: number of neighbor window to extract
- weight w inside of the positive unlabeling

Evaluation
- sliding padding
- delay tolerance

## Ideas
idea random: nella loss mettere contributo dato dalla discriminazione degli encoding del matrix profile? 


## Results on Yahoo dataset (splitted half/half) 

####  ENCODING SIZE 160 | SLIDING WINDOW 30 | ALPHA 0.5 | HIDDEN SIZE 100 | mp window mean

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

####  ENCODING SIZE 160 | SLIDING WINDOW 30 | ALPHA 0.5 | HIDDEN SIZE 100 | MP SUM


********** Results for  KNN
Anomaly detection AUC:  0.5031878884623932
Anomaly detection AUPRC:  0.14390457115938712
Label 0:  0.3206629212859704 +- 0.4349070794853691
Label 1:  0.31950838786197905 +- 0.42106749455307535
********** Results for  LOF
Anomaly detection AUC:  0.507518646691755
Anomaly detection AUPRC:  0.14651789907622703
Label 0:  4.4288654 +- 92.65485
Label 1:  3.185636 +- 70.77686
********** Results for  CBLOF
Anomaly detection AUC:  0.4983150687957835
Anomaly detection AUPRC:  0.1438732971098424
Label 0:  1.3792383833049726 +- 1.219904295697673
Label 1:  1.3812950035848577 +- 1.2304679271942904
********** Results for  MCD
Anomaly detection AUC:  0.5040601861722276
Anomaly detection AUPRC:  0.14725469881690342
Label 0:  44806752208.541115 +- 122178367232.16676
Label 1:  47766392098.83831 +- 127649934852.30925
********** Results for  PCA
Anomaly detection AUC:  0.5018953803844088
Anomaly detection AUPRC:  0.14638809493646415
Label 0:  4.016528993320838e+17 +- 1.2318585028385149e+17
Label 1:  4.043978131461425e+17 +- 1.279975983592105e+17


``
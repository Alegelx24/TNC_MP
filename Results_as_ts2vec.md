# Results on Yahoo dataset (splitted half/half ON EACH SUBSEQUENCE) 

# BASELINE RESULTS (best #### ENCODING 40 | Mc 20 | window 4 | n9)

### RESULTS WITH DIFFERENT WINDOW SIZE ( 4 - 30 - 60 - 120 - 672 )
#### ENCODING 160 | Mc 20 | window 4 | n1




#### ENCODING 160 | Mc 20 | window 30 | n2




#### ENCODING 160 | Mc 20 | window 60 | n3



#### ENCODING 160 | Mc 20 | window 120 | n4


#### ENCODING 160 | Mc 20 | window 48 | n5

### RESULTS WITH DIFFERENT ENCODING SIZE (640 - 320 - 160(upon) - 80 - 40)

#### ENCODING 320 | Mc 20 | window 4 | n6

#### ENCODING 640 | Mc 20 | window 4 | n7

#### ENCODING 80 | Mc 20 | window 4 | n8

#### ENCODING 40 | Mc 20 | window 4 | n9


#### ENCODING 10 | Mc 20 | window 4 | n10


### RESULTS WITH DIFFERENT MC SAMPLE SIZE (40 - 20 - 10)

#### ENCODING 40 | Mc 40 | window 4 | n11

#### ENCODING 40 | Mc 20 | window 4 | n9 upon

#### ENCODING 40 | Mc 10 | window 4 | n12



# MP hybrid penalty RESULTS (with ENCODING 40 | Mc 20 | window 4 | n9) and ( DAMP 672-4032 )

## it depends on Alpha parameter (0.1 - 0.3 - 0.5 - 0.7 - 0.9)

#### ALPHA 0.1 | n13

#### ALPHA 0.3 | n14



#### ALPHA 0.5 | n15

#### ALPHA 0.7 | n16

#### ALPHA 0.9 | n17

#### ALPHA 0.95 | n18

#### ALPHA 0.85 | n19

#### ALPHA 0.80 | n20

# MP hybrid penalty(SUM LOSS) RESULTS (with ENCODING 40 | Mc 20 | window 4 | n9) and ( DAMP 672-4032 )

## Alpha parameter (0.1 - 0.3 - 0.5 - 0.7 - 0.9)

#### ALPHA 0.1 | n21

#### ALPHA 0.3 | n22


#### ALPHA 0.5 | n23

#### ALPHA 0.7 | n24

#### ALPHA 0.9 | n25

#### ALPHA 0.6 | n26

#### ALPHA 0.4 | n27

# MP hybrid penalty(Sum only top k discord) RESULTS (with ENCODING 40 | Mc 20 | window 4 | n9) and ( DAMP 672-4032 )

## Alpha parameter (0.1 - 0.3 - 0.5 - 0.7 - 0.9), K=1

#### ALPHA 0.1 | n28

#### ALPHA 0.3 | n29

#### ALPHA 0.5 | n30

#### ALPHA 0.7 | n31

#### ALPHA 0.9 | n32

# MP hybrid penalty with MP encoding discrimination (ENCODING 40 | Mc 20 | window 4) and ( DAMP 672-4032 )

#### ALPHA 0.9 | n33

# Enhancing Continual Deep Open Set Recognition with Perceptive Unknown Feature Search - Class Incremental Quad-Channel constrastive Prototype Networks (CI-QCCPN).

This is the official code for the paper entitled "Enhancing Continual Deep Open Set Recognition with Perceptive Unknown Feature Search"

![MarineGEO circle logo](/ci_qccpn.jpg)

Note that in the paper:

* In Section II-C, the PUFS process is not presented in step 2, but rather in **step 3**.
* In Section III, we run the model with **three permutations** instead of five permutations.

## Feature Extraction
If you do not have the dataset, please download the extracted features from this link: https://drive.google.com/file/d/1zZUGH3J4K-8R0wPR-erwdL2DNJMEoKCz/view?usp=sharing

Otherwise, you can run the code by:

```
python main.py extract_feature --dataset uecfood100 --datapath FOLDER_CONTAINING_FOLDERS_OF_DATASET
```

The supported datasets for feature extraction are oxfordpet and uecfood100.

## Rejection Evidence

```
python main.py reject_forget_evidence --dataset uecfood100 --n_epoch 50 --model linear_softmax --trial 5
```

The supported methods are linear_softmax and qccpn

## CI-QCCPN

For running CI-QCCPN, you can use this argument
```
python main.py ci_qccpn --dataset uecfood100 --n_epoch 50 --trial 5
```

Please check "_result" folder for the results of the running code.

Please feel free to contact me if you encounter any issues :).

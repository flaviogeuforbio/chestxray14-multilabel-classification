
# NIH ChestXRay14 – Multi-label Classification + Grad-CAM

End-to-end pipeline for **multi-label chest X-ray classification** on the NIH ChestXRay14 dataset using a fine-tuned ResNet50.

The project focuses not only on ranking performance (ROC-AUC), but also on **decision threshold calibration**, **error analysis**, and **qualitative interpretability** via Grad-CAM.

## Dataset
- Dataset: **NIH ChestXRay14** (~112k images, Data_Entry_2017.csv + train_val/test lists)
- Task: multi-label classification (an image can contain multiple findings)

**Note**: labels are derived from radiology reports and are known to be noisy; this **strongly impacts** calibration and the reliability of hard binary decisions.
## Highlights

- Multi-label setup (14 pathologies) with **Binary Cross-Entropy Loss** per class and **positional weights** for class imbalance
- Fine-tuning strategy: **ResNet50** (pretrained on ImageNet) with layer 1, 2, 3 *freezed*, fine-tuning on layer 4 (*starting lr*: 1e-4), custom head classifier trained (*starting lr*: 1e-3)
- Validation model selection via **mean ROC-AUC**
- **Grad-CAM** heatmaps computed per selected class for qualitative inspection


## Evaluation

### Metrics

- Primary metric: **mean ROC-AUC** over 14 classes
- Achieved: **~0.72–0.73** on validation set (more detailed results below)  

#### Why ROC-AUC is not reliable for imbalanced data
ROC-AUC measures ranking (how often positives are scored above negatives) and does not guarantee calibrated probabilities or clean separation for a single threshold—especially under strong class imbalance and label noise.

#### Threshold calibration
Raw probabilities are typically low and not directly usable with a fixed standard 0.5 threshold.
To produce actionable binary outputs, I calibrated **class-specific thresholds** on the validation set by maximizing F1 per class, and report in inference:

- raw probability *p*
- class threshold *t*
- POS/NEG prediction
- decision margin (absolute and relative)

This separates:

- **model scoring** (output of the network)
- **decision policy** (interpretation of the predictions)

Thresholds and metrics are saved in a structured JSON artifact.



## Qualitative Analysis

The project includes:

- **Per-class score distribution analysis** (positives vs negatives) with box plots & violin plots (using ```validate.py```)
- **Grad-CAM heatmaps** computed per class, selected according to calibrated decision margins (using ```inference.py```)

Grad-CAM is used qualitatively, not as a localization ground truth.

![Score distribution for Atelectasis](prob_analysis/dist_Atelectasis.png)

![Grad-CAM example for Infiltration](gradcam_analysis/gradcam_Infiltration.png)


## Results

The following table shows the calculated metrics (ROC-AUC/F1/PR) and best estimated threshold for 6 choosen classes (2 common, 2 intermediate, 2 rare):

| Class | Test ROC-AUC | Val Threshold | Val F1 | Val Precision | Val Recall |
|------|--------------|--------------|--------|---------------|------------|
| Infiltration | 0.67 | 0.17 | 0.35 | 0.26 | 0.55 |
| Effusion | 0.78 | 0.22 | 0.49 | 0.42 | 0.60 |
| Emphysema | 0.84 | 0.24 | 0.33 | 0.35 | 0.31 |
| Cardiomegaly | 0.80 | 0.26 | 0.25 | 0.25 | 0.25 |
| Fibrosis | 0.78 | 0.19 | 0.16 | 0.18 | 0.14 |
| Hernia | 0.80 | 0.05 | 0.28 | 0.5 | 0.19 |

**Important Note**: Despite decent ROC-AUC values, F1 scores remain modest due to heavy overlap between positive and negative score distributions.

## Limitations

- **Label noise** limits instance-level reliability
- **Strong class imbalance** leads to low calibrated probabilities
- ROC-AUC does not imply usable binary decisions
- Grad-CAM may highlight **confounders** or diffuse regions

This behavior is **expected** for ChestXRay14 and widely reported in the literature.
## How to Run

### Importing the project

Clone the project

```bash
  git clone https://github.com/flaviogeuforbio/chestxray14-multilabel-classification
```

Go to the project directory

```bash
  cd chestxray14-multilabel-classification
```

Change branch if used on Kaggle dataset structure

```bash
  git checkout kaggle-dataset-layout
```

### Training

```bash
  python src/train.py \
    --data_root <DATA_ROOT> \
    --checkp_dir <CHECKPOINT_DIR> \
    --n_epochs <EPOCHS> \
    --patience <PATIENCE> (optional)
```

- data_root: NIH ChestXRay14 dataset folder
- checkp_dir: Checkpoint models folder
- n_epochs: Number of training iterations
- patience: Early stopping parameter

### Evaluating on test set

```bash
  python src/validate.py \
    --data_root <DATA_ROOT> \
    --checkpoint <CHECKPOINT_PATH> \
    --per_class (optional)
```

- checkpoint: Path to checkpoint model to test 
- per_class: if True ROC-AUC score per class + graphical analysis (box/violin plots) per class are returned

### Calibrating thresholds on validation set

```bash
  python src/calibrate_thresholds.py \
    --data_root <DATA_ROOT> \
    --checkpoint <CHECKPOINT_PATH> \
```

Thresholds and metrics saved in ```thresholds.json```.

### Inference + GradCAM

```bash
python inference.py \
  --image <IMAGE_PATH> \
  --checkpoint <PATH> \
  --thresholds thresholds.json \
  --gradcam (optional)
```
It needs ```thresholds.json``` previously created with ```calibrate_thresholds.py```.

- image: Path to image to use for Inference
- gradcam: if True GradCAM heatmaps are shown for most eligible classes (according to decision margins)

## Takeaway

This project highlights a critical lesson in applied medical AI:

*Good ranking performance does not imply reliable inference*.

By explicitly addressing calibration, thresholds, and interpretability, the pipeline demonstrates how to reason about model outputs **beyond headline metrics**.
## Next Steps

- Probability calibration (Platt scaling / isotonic regression)
- Architectures specialized for medical imaging
- Cleaner or curated datasets
- Shift from classification to localization or segmentation tasks
## Final notes

This project is intentionally documented with transparency about its limitations and failure modes, reflecting a realistic applied machine learning workflow.
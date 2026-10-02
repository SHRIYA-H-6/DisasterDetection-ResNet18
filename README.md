# DisasterDetection-ResNet18

Disaster Detection from Satellite Images: second model, built on **ResNet-18**.
Neural Networks and Deep Learning (23CSE473) course project, Team A1 (CSE A).

Given a pair of pre- and post-event satellite images, the system:

1. classifies the **disaster type** (Earthquake, Tsunami, Hurricane) and **severity** (Low, Moderate, Severe), giving 9 combined classes;
2. segments the **damaged buildings** with a ResNet-18 U-Net and measures the **affected area %** from that mask;
3. draws the affected regions on a **Matplotlib overlay** and an interactive **Folium map** at real coordinates;
4. writes an automated **summary report** with a recommended emergency action.

All images are real. No synthetic or placeholder data is used anywhere.

## Data

| Disaster type | Source | Events | Post-event image |
|---|---|---|---|
| Earthquake | xBD (Gupta et al., 2019) + BRIGHT (Chen et al., 2025) | `mexico-earthquake` + BRIGHT `*earthquake*` events | optical (xBD), SAR (BRIGHT) |
| Tsunami | xBD | `palu-tsunami` (and `sunda-tsunami` if present) | optical |
| Hurricane | xBD | `hurricane-michael` (and `-harvey`, `-florence`, `-matthew` if present) | optical |

Severity is the share of damaged buildings in the tile (xBD: minor/major/destroyed buildings among classified buildings; BRIGHT: damaged/destroyed building pixels): Low ≤ 10% < Moderate ≤ 30% < Severe.

xBD's only earthquake (Mexico) has no moderately or severely damaged tiles, so Earthquake Moderate/Severe come from BRIGHT.

The datasets cannot be redistributed, so they live in **private** data repos cloned next to this one:

```
../xbd-data      mexico-earthquake + palu-tsunami (xBD train set) + xview_geotransforms.json.tgz
../xbd-data-2    hurricane-michael (xBD train set) + bright/ (BRIGHT earthquake subset)
```

Every sibling folder named `xbd-data*` is used automatically. Override with `DISASTER_DATA_ROOTS` (`:`-separated on Linux/macOS, `;` on Windows). To build `xbd-data-2` from the official downloads:

```
python scripts/prepare_xbd_subset.py    <xview2 train tar.gz> ../xbd-data-2    # hurricane-michael by default
python scripts/prepare_bright_subset.py ../xbd-data-2 <BRIGHT zip(s) or folder>
```

## Models

* **Classifier:** pseudo-Siamese multitask ResNet-18. Two separate ImageNet-pretrained ResNet-18 encoders (pre / post). Their pooled 512-d features and the difference between them feed a dropout-regularised shared layer, then two softmax heads (type, severity). The loss is the sum of the two cross-entropies, with equal weights. Optimiser: Adam.
* **Segmentation:** U-Net with a shared-weight (Siamese) ImageNet-pretrained ResNet-18 encoder. At each scale, the pre features, post features and their difference are fused into the skip connection. It predicts background / intact building / damaged building per pixel, and is trained on the datasets' pixel-level labels. Affected Area % = damaged-building pixels / all pixels of the predicted mask.

ImageNet weights: timm `resnet18.a1_in1k`, downloaded from the timm GitHub release. The SHA-256 is checked on download.

Overfitting safeguards (classifier; most also apply to the U-Net):

* dropout (0.5) in the heads (0.2 in the deep U-Net decoder blocks);
* Adam weight decay (L2);
* flip / rotation / brightness-contrast augmentation on the training split only;
* class-balanced sampling;
* early stopping on validation loss (patience 7) with the best weights restored;
* progressive unfreezing of the encoders: heads only, then `layer4` (epoch 4), `layer3` (8), `layer2` (12), everything (16);
* an explicit overfitting detector that stops training when validation loss rises while training loss falls;
* stratified k-fold cross-validation;
* the train/val loss gap reported in the final evaluation.

## Running

```
pip install -r requirements.txt
python -m disaster_detection.build_metadata        # stage 1: labels, splits, minimum-sample check (>= 10 per class)
python -m disaster_detection.preprocess            #          cache 256px tiles
python -m disaster_detection.train_classifier --cv-folds 5
python -m disaster_detection.train_segmentation
python -m disaster_detection.evaluate              # 3 reports (console + JSON), confusion matrices, CSV
python -m disaster_detection.demo                  # end-to-end on real test images -> report + overlay + map
python -m disaster_detection.predict --pre a.png --post b.png   # any new image pair
```

Outputs go to `outputs/` (override with `DISASTER_OUTPUT_ROOT`).

## Known limitations

* BRIGHT post-event images are SAR, while xBD post-event images are optical. The type head can partly tell BRIGHT earthquakes apart by modality. `evaluate` therefore reports accuracy per source and modality (`source_breakdown.csv`), so this effect is visible.
* Severity measures building damage. "Affected area" is the area of damaged buildings, not of floodwater or wind-swept land.

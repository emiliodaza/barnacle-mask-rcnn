# Counting barnacles with Mask R-CNN

This is my prototype for the DALI Data Challenge 26F. The goal is to help count barnacles inside a sampling frame by producing separate predicted masks that can be checked over the original photograph.

The [notebook](barnacle_experiment.ipynb) follows the experiment from inspecting the blue annotations to reviewing the model's predictions. The saved figures are included, so the results can be read without access to Discovery or a GPU.

## What I tried

I started by checking how the supplied contours line up with the photographs. I sampled blue annotation pixels in HSV color space, extracted those lines, and used connected components to recover the regions enclosed by them. These regions became candidate instance labels. This recovery step is automatic, so I treat the labels as provisional.

I then used sliders to mark the inside of the sampling frame. The selected corners are saved in original-image coordinates and applied to both the photograph and its labels. Image 1 supplies training data; image 2 is kept for validation. The recovered arrays are saved separately so they can be loaded on the GPU without processing the contours again.

For the model, I fine-tuned Torchvision's COCO-pretrained Mask R-CNN with a ResNet-50 backbone and Feature Pyramid Network. The experiment uses small-object anchors, horizontal flips, mixed precision, and 200 training updates. After training, I compared nine combinations of detection-confidence and mask thresholds on image 2, selected by instance F1 at mask IoU 0.5. This is an inference-threshold search; I did not run a learning-rate or training-duration sweep.

## What the results show

The validation comparison uses 410 recovered reference instances after resizing, down from 426 before resizing. Both counts exclude the single-pixel candidates removed during preprocessing.

| Setting | Confidence | Mask threshold | Precision | Recall | F1 | Predicted count | Count error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline thresholds | 0.5 | 0.5 | 0.091 | 0.137 | 0.109 | 614 | 204 |
| Selected by F1 | 0.3 | 0.5 | 0.094 | 0.232 | 0.134 | 1,013 | 603 |

Lowering the confidence threshold improved recall and F1, but the count moved further from the recovered reference total. Choosing thresholds by instance matching therefore revealed a trade-off with count accuracy. Agreement with the reference masks remains weak. The [full threshold search](outputs/validation_tuning.csv) and [validation summary](outputs/validation_summary.json) are saved with the figures.

![Validation photograph, predicted masks, and overlay](outputs/validation_prediction.png)

Unseen image 1 is the most encouraging visual result. Many predicted masks follow individual barnacle interiors across the dense clusters, while the large exposed-rock area receives few predictions. This suggests that useful visual cues carried over to a photograph the model had not trained on. Its predicted count is 1,039; without reference masks, I cannot verify that count.

![Predictions on the wider unseen photograph](outputs/unseen_img1_prediction.png)

The additional close-up is a useful failure case. Some individual barnacles receive several small, partial predictions, making its predicted count of 864 unreliable. The result suggests sensitivity to scale and appearance; this experiment does not isolate perspective as the cause.

![Fragmented predictions on the additional close-up](outputs/unseen_img2_prediction.png)

## Reading or rerunning the notebook

The model ran as Python code in a terminal session on Dartmouth Discovery. I copied its outputs back to the local project and split the model code into notebook cells. The **Results** section reads those saved files; it does not run training again.

### Looking at the existing results

You can read the notebook and figures on GitHub. For a live local notebook, use Python 3.10 for the dependency versions in this repository. From the project root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook barnacle_experiment.ipynb
```

Run only the code cell under **Results** to display the saved comparison and figures. It does not require CUDA. The earlier inspection and region-selection sections can also run locally; their sliders need a live kernel and will not respond in GitHub's static preview.

### Running the model again

The recorded GPU run used Python 3.10, PyTorch 2.1.2+cu118, Torchvision 0.16.2+cu118, and an NVIDIA A100 MIG allocation with about 40 GB of GPU memory. The code can run in another compatible NVIDIA GPU environment; requesting access to a particular cluster is outside this notebook.

In a fresh Python 3.10 GPU environment, install the recorded CUDA 11.8 PyTorch build first, then the remaining requirements. Use this route instead of the local installation step above:

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

The command follows the [official PyTorch instructions for version 2.1.2](https://pytorch.org/get-started/previous-versions/#v212).

From the project root, run the cells from **Model experiment** through **Predicting both unseen photographs** in order, in the same CUDA-enabled Python process. You can use a GPU notebook kernel or combine the Python code cells into one terminal run, as I did on Discovery. They load the saved masks and both region files directly, so the interactive preparation does not need to be repeated. The model downloads its pretrained weights on first use if they are not already cached.

The run writes figures, the threshold-search CSV, and JSON summaries to `outputs/`, replacing the corresponding files. Run the **Results** cell afterward to view them. Random seeds are fixed, but the saved predictions may not reproduce exactly across different hardware and library builds.

### Revisiting the labels or selections

The earlier notebook sections explain how the labels were recovered and how the frame corners were selected. Their controls start from the recorded inspection position and saved corners. If you move the inspection crop, choose new blue sample pixels before recomputing the color bounds. If you change a counting region, regenerate the results that depend on it.

## Where things are saved

| File or folder | What it contains |
| --- | --- |
| [`barnacle_experiment.ipynb`](barnacle_experiment.ipynb) | The full investigation, commented code, saved results, and conclusions. |
| [`Barnacles/`](Barnacles/) | Photographs, contour images, and the supplied cropped overlay used for inspection. The close-up is an additional unlabeled example. |
| [`configs/regions.json`](configs/regions.json) | The training/validation split, original dimensions, and selected frame corners. |
| [`configs/test_regions.json`](configs/test_regions.json) | The selected regions for both unseen photographs. The close-up uses the whole image. |
| [`data/processed/instances.npz`](data/processed/instances.npz) | Recovered label maps under `img1` and `img2`. Zero is background; positive IDs identify candidate interiors. |
| [`outputs/`](outputs/) | The loss curve, validation search, prediction figures, counts, and inference settings from the Discovery run. |
| [`requirements.txt`](requirements.txt) | Dependencies for installing the notebook in Python 3.10. |

## What I learned and would check next

The annotation recovery turned out to be part of the modeling problem itself. A region enclosed by blue pixels is a candidate label, and mistakes at that stage affect both training and evaluation. I would audit those masks before drawing stronger conclusions about the model.

The loss curve and overlays also answer different questions. Falling training loss shows adaptation to the training example, while the close-up makes the limits of that adaptation visible. More varied object scales and independently annotated photographs would be useful next experiments.

For now, the counts need human review. With one training photograph and one validation photograph, this is an exploratory prototype. The validation image was also used to select the thresholds, so its metrics are not an independent test score. A practical follow-up would measure whether checking and correcting the predictions takes less time than counting manually.

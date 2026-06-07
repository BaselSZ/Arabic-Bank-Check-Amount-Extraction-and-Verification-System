# Arabic Bank Check Amount Extraction & Processing

## 1. Problem Definition
Automatic processing of bank checks is an important task in financial systems because manual verification is time-consuming, expensive, and prone to human error. Arabic bank checks present additional challenges due to handwritten text variability, Arabic script complexity, and differences in writing styles.

The goal of this project is to build a system capable of:
1. Detecting the legal amount region and courtesy amount region from Arabic bank checks.
2. Recognizing the courtesy amount (numeric amount).
3. Recognizing the legal amount (Arabic text amount).
4. Verifying whether both recognized amounts correspond to the same monetary value.

### Project Breakdown
* **Part A:** Object detection for legal and courtesy amount regions.
* **Part B:** Courtesy amount recognition using OCR.
* **Part C:** Legal amount recognition using OCR.
* **Part D:** Verification between legal and courtesy amounts.

---

## 2. Dataset

### Dataset Structure
The dataset consists of scanned Arabic bank checks along with:
* **Bounding box annotations (YOLO-format):**
  * Legal amount region
  * Courtesy amount region
* **Ground-truth text files:**
  * Courtesy amounts (numeric values)
  * Legal amounts (Arabic text)

```
CheckImages/       --> Original check images
BoundingBoxes/     --> YOLO-format bounding box annotations
CourtesyAmounts/   --> Ground-truth numeric amounts
LegalAmounts/      --> Ground-truth Arabic text amounts
```

### Data Preparation

#### Part A
The dataset was converted into YOLO format and split into:
* **80%** Training
* **20%** Validation

A YAML configuration file was created containing dataset paths, number of classes, and class names:
| Class ID | Class Name |
| :---: | :--- |
| 0 | legal_amount |
| 1 | courtesy_amount |

#### Part B and Part C (OCR Tasks)
1. Bounding boxes were used to crop regions from the original checks.
2. Cropped images were saved separately.
3. Labels were cleaned and normalized.
4. Data augmentation techniques were applied.

---

## 3. Methodology

### Part A: Region Detection
* **Objective:** Detect the locations of the legal amount region and courtesy amount region.
* **Model:** YOLOv8 Nano (`yolov8n.pt`) for object detection.
* **Preprocessing:** TIFF images were converted and re-saved using OpenCV. Images and labels were organized into the YOLO directory structure.

#### Training Configuration
| Parameter | Value |
| :--- | :--- |
| Model | YOLOv8n |
| Epochs | 30 |
| Image Size | 640 |
| Batch Size | 16 |

* **Evaluation:** Evaluated using Mean IoU and Accuracy at thresholds $IoU \ge 0.50$, $IoU \ge 0.75$, and $IoU \ge 0.90$. Worst-case predictions were visualized for error analysis.

---

### Part B: Courtesy Amount Recognition
* **Objective:** Recognize handwritten numeric amounts from courtesy amount regions.
* **Preprocessing:** Courtesy amount crops were converted to grayscale, resized to $(32 \times 128)$, and augmented using random rotation and brightness/contrast jitter.
* **OCR Architecture:** A CRNN (Convolutional Recurrent Neural Network) architecture consisting of:
  1. CNN feature extractor
  2. Bidirectional LSTM sequence model
  3. CTC loss for alignment-free training
* **Character Set:** Only digits were used (0–9).

#### Training Configuration
| Parameter | Value |
| :--- | :--- |
| Epochs | 40 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss | CTC Loss |
| Batch Size | 32 |

* **Evaluation Metrics:** Digit-level accuracy and the percentage of amounts with no errors, one error, or two or more errors (evaluated via edit distance).

---

### Part C: Legal Amount Recognition
* **Objective:** Recognize Arabic handwritten legal amounts from checks.
* **Preprocessing:** Legal amount crops were converted to grayscale, resized to $(64 \times 512)$, and augmented using random rotation and random affine transformations. Arabic text normalization and cleaning were applied to remove noise and formatting artifacts.
* **OCR Architecture:** Another CRNN model used with a larger recurrent layer due to the increased complexity of Arabic text. It contains:
  1. CNN feature extractor
  2. Bidirectional LSTM
  3. Fully connected classification layer
  4. CTC decoding
* **Vocabulary:** Automatically constructed from all unique characters appearing in the dataset.

#### Training Configuration
| Parameter | Value |
| :--- | :--- |
| Epochs | 80 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Scheduler | ReduceLROnPlateau |
| Loss | CTC Loss |
| Batch Size | 16 |

* **Evaluation Metrics:** Character Error Rate (CER) and Word Error Rate (WER) using edit distance.

---

### Part D: Verification System
* **Objective:** Verify whether the courtesy amount and the legal amount represent the same monetary value.
* **Method:**
  1. Convert recognized courtesy amount digits into Arabic words.
  2. Normalize both the expected legal amount and the OCR-predicted legal amount.
  3. Remove common noise tokens.
  4. Compute edit-distance similarity.
  5. Accept checks if similarity $\ge 50\%$.
* **Verification Output:** Each check was labeled as `VERIFIED` or `FAILED`. Final verification accuracy was computed.

---

## 4. Experiments and Results

### Part A: Detection Results
The YOLOv8 model successfully detected both the legal amount and courtesy amount regions with good localization accuracy.

#### Quantitative Results
| Metric | Legal Amount | Courtesy Amount |
| :--- | :---: | :---: |
| **Mean IoU** | 78.13% | 70.48% |
| **Accuracy @ $IoU \ge 0.50$** | 91.39% | 82.22% |
| **Accuracy @ $IoU \ge 0.75$** | 74.44% | 52.78% |
| **Accuracy @ $IoU \ge 0.90$** | 18.61% | 10.83% |

* **Observations:** The detector achieved strong performance overall, particularly at the 0.50 IoU threshold, where both classes exceeded 80% accuracy. Legal amount regions achieved better localization accuracy than courtesy amount regions across all thresholds.

---

### Part B: Courtesy Amount Recognition Results
The CRNN-based OCR model achieved strong performance in recognizing handwritten courtesy amounts from Arabic bank checks.

#### Quantitative Results
| Metric | Result |
| :--- | :---: |
| **Digit-Level Accuracy** | 93.86% |
| **Amounts with No Errors** | 80.95% |
| **Amounts with One Error** | 14.57% |
| **Amounts with Two or More Errors** | 4.48% |

* **Observations:** The model achieved a high digit-level accuracy of 93.86%, indicating that most handwritten digits were recognized correctly. Additionally, more than 80% of the courtesy amounts were predicted perfectly without any recognition errors. Most incorrect predictions differed from the ground truth by only a single digit (14.57% one-error rate). Only a small percentage of samples contained two or more errors.

---

### Part C: Legal Amount Recognition Results
The legal amount recognition task was considerably more challenging than courtesy amount recognition due to the complexity of Arabic handwritten text and the larger vocabulary size.

#### Quantitative Results
| Metric | Result |
| :--- | :---: |
| **Character Error Rate (CER)** | 23.00% |
| **Word Error Rate (WER)** | 35.40% |

* **Observations:** The CRNN model achieved a Character Error Rate (CER) of 23.00%, indicating that most characters were recognized correctly even in difficult handwritten samples. However, the Word Error Rate (WER) was significantly higher at 35.40%, showing that errors at the character level often propagated into full-word recognition mistakes. The gap between CER and WER reflects the difficulty of Arabic handwritten OCR, where even a small character substitution can completely alter the meaning of a word.

---

### Part D: Verification Results
The final verification stage compared the recognized courtesy amount with the recognized legal amount to determine whether both represented the same monetary value.

#### Quantitative Results
| Metric | Result |
| :--- | :---: |
| **Verified Checks** | 227 |
| **Failed Checks** | 62 |
| **Verification Accuracy** | 78.55% |

* **Observations:** The verification module achieved an overall accuracy of 78.55%, successfully validating the majority of bank checks despite OCR imperfections. The verification process proved robust against minor OCR mistakes because it relied on normalized edit-distance similarity rather than exact string matching. As a result, many checks were correctly verified even when small character substitutions or missing letters were present in the OCR output.

---

## 5. Discussion and Error Analysis
Removed for privacy reasons

---

## 6. Conclusions
This project implemented a complete Arabic bank check processing pipeline including region detection, numeric OCR, Arabic text OCR, and amount verification. 

YOLOv8 successfully localized both legal and courtesy amount regions, while CRNN-based OCR models were able to recognize both numeric and Arabic handwritten text with reasonable accuracy. The final verification stage demonstrated that combining numeric and textual recognition can improve reliability and help identify inconsistencies between check fields.

---

## 7. Limitations
1. Performance depends heavily on handwriting quality and annotation quality.
2. Arabic legal amount recognition remains highly challenging.
3. The dataset size is relatively limited.
4. Verification relies on heuristic similarity thresholds.
5. Some Arabic linguistic variations are not fully handled.

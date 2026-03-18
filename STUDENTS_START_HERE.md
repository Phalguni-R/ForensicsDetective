# Assignment 2: Getting Started

**Course:** EAS 510 - Basics of AI
**Assignment:** ForensicsDetective - Hero or Zero?

---

## Your Starting Point

You do **NOT** need to create PDFs or run any conversion scripts. The PNG images are already provided for you.

### Use These Folders

| Folder | Contents | Count |
|--------|----------|-------|
| `word_pdfs_png/` | Word-generated PDF images | 398 |
| `google_docs_pdfs_png/` | Google Docs-generated PDF images | 396 |
| `python_pdfs_png/` | Python/ReportLab-generated PDF images | 100 |

**Total: ~894 images ready for your augmentation and analysis work.**

---

## Assignment Workflow

### Task 1: Repository Setup (10 pts)
1. Fork this repository to your GitHub account
2. Clone your fork locally
3. Add `delveccj` and `AnushkaTi` as collaborators
4. Create your `SETUP.md` documenting your environment setup

### Task 2: Dataset Augmentation (25 pts)
Apply **5 augmentations** to the existing PNG images:
- Gaussian Noise (σ ∈ [5, 20])
- JPEG Compression (quality 20-80)
- DPI Downsampling (300 → 150 or 72)
- Random Cropping (1-3% border removal)
- Bit-Depth Reduction (8-bit → 4-bit)

**Result:** Your augmented dataset should be **6× the original size** (1 original + 5 augmented per image).

### Task 3: Robustness Testing (20 pts)
- Train classifiers on **original images only**
- Test on each augmentation type separately
- Measure accuracy degradation

### Task 4: Additional Classifiers (15 pts)
Add **2 classifiers** from different categories (e.g., Random Forest + MLP, or XGBoost + CNN).

### Task 5: Comprehensive Analysis (20 pts)
- Compute accuracy, precision, recall, F1
- Generate confusion matrices
- Perform statistical significance testing

### Task 6: Research Report (10 pts)
Write an 8-12 page report following the structure in the assignment PDF.

---

## Relevant Scripts (Reference Only)

These scripts show how the original conversion/classification was done. You may reference them but focus your work on augmentation and new classifiers:

| Script | Purpose | Do You Need It? |
|--------|---------|-----------------|
| `pdf_to_binary_image.py` | Converts PDFs to PNG | No - already done |
| `train_*_classifiers.py` | Example classifier training | Reference only |
| `create_comparison_images.py` | Visualization | Optional reference |

---

## File Organization for Submission

```
Assignment2_YourName/
├── README.md
├── SETUP.md
├── data/
│   ├── original_images/      # Copy or link to *_pdfs_png folders
│   └── augmented_images/     # Your augmented outputs
├── src/
│   ├── augmentation.py       # Your augmentation code
│   ├── classification.py     # Your classifiers
│   └── analysis.py           # Your analysis pipeline
├── results/
│   ├── confusion_matrices/
│   ├── robustness_plots/
│   └── performance_metrics.csv
├── reports/
│   └── final_research_report.pdf
└── requirements.txt
```

---

## Quick Start Checklist

- [ ] Fork and clone the repository
- [ ] Verify you can access the PNG images in `*_pdfs_png/` folders
- [ ] Set up your Python environment (see `requirements.txt` or create your own)
- [ ] Create `SETUP.md` with your setup steps
- [ ] Begin implementing augmentations in `src/augmentation.py`

---

## Questions?

If anything is unclear, reach out to the instructor or TAs before the deadline.

**Due Date:** March 29, 2026 at 11:59 PM

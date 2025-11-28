# GeoLife Next-Location Prediction System

**A hierarchical Transformer-based deep learning system for predicting the next location in human mobility trajectories.**

## 🎯 Performance

- **Test Accuracy@1:** 42.65%
- **Test Accuracy@5:** 60.86%  
- **Test MRR:** 51.01%
- **Model Parameters:** 411,123 (< 500K budget)

## 📚 Complete Documentation

**For comprehensive, detailed documentation covering every aspect of this implementation, please see:**

### [📖 COMPREHENSIVE_DOCUMENTATION.md](COMPREHENSIVE_DOCUMENTATION.md)

This 1,700+ line document is the **single source of truth** and includes:

- ✅ **Complete architecture explanation** with diagrams
- ✅ **Step-by-step implementation details**  
- ✅ **Full data pipeline documentation**
- ✅ **Training methodology and hyperparameters**
- ✅ **Evaluation metrics with code**
- ✅ **Usage guide and examples**
- ✅ **Troubleshooting and debugging tips**
- ✅ **Future improvement roadmap**

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_v2.py

# Evaluate on test set
python test.py
```

## 📁 Project Structure

```
expr_hrcl_next_pred_av6/
├── COMPREHENSIVE_DOCUMENTATION.md  ← Read this!
├── src/
│   ├── models/multitask_transformer.py  (Best model)
│   ├── data/dataset.py
│   ├── training/multitask_trainer.py
│   └── ...
├── data/geolife/
├── checkpoints/
└── logs/
```

## 🔬 Research Highlights

- **Hierarchical spatial encoding** with H3 and S2 geospatial indices
- **Multi-resolution features** (8 spatial levels + temporal + user)
- **Parameter-efficient Transformer** architecture
- **Proper train/val/test splits** (no data leakage)
- **Full GPU acceleration** with PyTorch

## 📊 Key Results

| Metric  | Value  |
|---------|--------|
| Acc@1   | 42.65% |
| Acc@5   | 60.86% |
| Acc@10  | 63.84% |
| MRR     | 51.01% |
| NDCG    | 54.28% |

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{geolife_next_loc_2025,
  title={Hierarchical Transformer for Next-Location Prediction},
  author={PhD-Style Research Project},
  year={2025},
  url={https://github.com/aetandrianwr/expr_hrcl_next_pred_av6}
}
```

## 📝 License

Research and educational use.

---

**For complete technical details, algorithms, and implementation guide:**  
**→ See [COMPREHENSIVE_DOCUMENTATION.md](COMPREHENSIVE_DOCUMENTATION.md) ←**

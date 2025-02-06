# Anomaly-Detection-with-Deep-SVDD

## Enhancing DeepSVDD with LeNet Architecture

---

## **Overview**
This project enhances the **Deep Support Vector Data Description (DeepSVDD)** framework by integrating the **LeNet architecture** to improve anomaly detection performance on **MNIST**, **CIFAR-10**, and adversarial detection in **GTSRB stop sign images**. The modifications optimize **efficiency, stability, and accuracy** while ensuring reproducibility.

---

## **Key Enhancements**

### **1. Architectural Modifications**
- **Kernel Size Reduction:** Decreased from **5×5 to 3×3** for improved computational efficiency.
- **Padding Adjustments:** Set to **1** to maintain spatial dimensions, ensuring consistency.
- **Batch Normalization:** Enabled learnable scaling and shifting (**affine=True**) for better generalization.
- **Transpose Convolution Adjustments:** Reduced kernel size to **3×3** with padding **1** for improved upsampling.
- **Explicit Upsampling:** Incorporated **F.interpolate** with a **scale factor of 2** for memory-efficient spatial upsampling.
- **Fully Connected Layer Normalization:** Added **BatchNorm1d** to stabilize activations and enhance training dynamics.

### **2. Mixed Precision Training**
- Dynamically selects **FP16** or **FP32** depending on the operation.
- Implemented using **GradScaler** in PyTorch to optimize GPU utilization.

---

## **Experimental Results**

### **1. Performance Improvement**
- The modified model achieved an **average ROC AUC of 59.55**, compared to **57.69** for the original on **CIFAR-10 (Cat class)** over 10 different seeds.
- **Paired t-test (p = 0.0002)** confirms statistically significant improvement.

### **2. Efficiency Gains**
- **Mixed Precision Training** reduced memory usage and training time by **~23%**.
- The modified model achieved an **average training time of 172.74**, compared to **225.51** for the original.
- **Paired t-test (p = 5.17×10⁻¹⁵)** confirmed statistical significance.

---

## **Conclusion**
The integration of **LeNet**, **architectural optimizations**, and **Mixed Precision Training** significantly enhanced DeepSVDD’s performance and efficiency. These improvements make the model more **robust, scalable, and reproducible** for **anomaly detection** in image datasets.

---

## **Acknowledgments**
This work is inspired by [**DeepSVDD**](https://proceedings.mlr.press/v80/ruff18a.html) and aims to improve its **efficiency and performance** for real-world applications.

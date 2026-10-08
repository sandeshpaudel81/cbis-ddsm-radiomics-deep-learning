| Model | Family | AUC (95% CI) | Sens. | Spec. | F1 | Acc. |
|---|---|---|---|---|---|---|
| DenseNet121 (frozen) + Radiomics | Early fusion | 0.782 (0.735–0.828) | 0.630 | 0.766 | 0.655 | 0.705 |
| DenseNet121 (frozen) | Deep | 0.775 (0.729–0.822) | 0.648 | 0.751 | 0.662 | 0.705 |
| ResNet50 (frozen) + Radiomics | Early fusion | 0.769 (0.717–0.815) | 0.620 | 0.762 | 0.647 | 0.699 |
| Late fusion: Swin-T fine-tuned + Radiomics | Late fusion | 0.769 (0.713–0.817) | 0.653 | 0.688 | 0.639 | 0.672 |
| Late fusion: Swin-T (frozen) + Radiomics | Late fusion | 0.760 (0.709–0.809) | 0.602 | 0.755 | 0.631 | 0.687 |
| ResNet50 (frozen) | Deep | 0.757 (0.705–0.803) | 0.597 | 0.747 | 0.625 | 0.680 |
| Swin-T fine-tuned (features) | Deep | 0.751 (0.699–0.800) | 0.602 | 0.729 | 0.621 | 0.672 |
| Swin-T fine-tuned + Radiomics | Early fusion | 0.749 (0.697–0.798) | 0.606 | 0.714 | 0.618 | 0.666 |
| DeiT-S (frozen) + Radiomics | Early fusion | 0.746 (0.690–0.797) | 0.648 | 0.732 | 0.654 | 0.695 |
| Swin-T fine-tuned (end-to-end) | Deep | 0.743 (0.686–0.795) | 0.671 | 0.651 | 0.637 | 0.660 |
| Swin-T (frozen) + Radiomics | Early fusion | 0.742 (0.689–0.794) | 0.606 | 0.740 | 0.628 | 0.680 |
| Radiomics | Radiomics | 0.736 (0.680–0.783) | 0.611 | 0.688 | 0.611 | 0.654 |
| EfficientNetV2-S (frozen) + Radiomics | Early fusion | 0.729 (0.674–0.777) | 0.579 | 0.743 | 0.610 | 0.670 |
| Swin-T (frozen) | Deep | 0.725 (0.672–0.778) | 0.574 | 0.747 | 0.608 | 0.670 |
| DeiT-S (frozen) | Deep | 0.721 (0.667–0.771) | 0.588 | 0.714 | 0.605 | 0.658 |
| EfficientNetV2-S (frozen) | Deep | 0.701 (0.644–0.752) | 0.597 | 0.688 | 0.601 | 0.647 |

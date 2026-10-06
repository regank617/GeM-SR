# Third-Party Notices

GeM-SR's own code is released under the MIT License (see [LICENSE](LICENSE)). This repository also
redistributes modified copies of the following third-party projects, which remain under their
original licenses. Each vendored copy keeps its upstream license file, and our changes are documented
in the `*_MODIFICATIONS.md` file next to it.

| Component | Location in this repo | Upstream | License | Our modifications |
|---|---|---|---|---|
| SwinIR | `benchmarks/swinir/repo/` | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) | Apache-2.0 (`benchmarks/swinir/repo/LICENSE`) | [SWINIR_MODIFICATIONS.md](benchmarks/swinir/SWINIR_MODIFICATIONS.md) |
| SR3 | `benchmarks/sr3/repo/` | [Janspiry/Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement) | Apache-2.0 (`benchmarks/sr3/repo/LICENSE`) | [SR3_MODIFICATIONS.md](benchmarks/sr3/SR3_MODIFICATIONS.md) |
| Fast Diffusion (Flow Matching) baseline | `benchmarks/fast_diffusion/` | Reuses the SR3 UNet above | Apache-2.0 (SR3 portions) | [FAST_DIFFUSION_MODIFICATIONS.md](benchmarks/fast_diffusion/FAST_DIFFUSION_MODIFICATIONS.md) |
| VARSR | `benchmarks/varsr/repo/` | [quyp2000/VARSR](https://github.com/quyp2000/VARSR) (built on [FoundationVision/VAR](https://github.com/FoundationVision/VAR)) | MIT (`benchmarks/varsr/repo/LICENSE`) | [VARSR_MODIFICATIONS.md](benchmarks/varsr/VARSR_MODIFICATIONS.md) |
| BasicSR (vendored inside VARSR) | `benchmarks/varsr/repo/basicsr/` | [xinntao/BasicSR](https://github.com/xinntao/BasicSR) | Apache-2.0 (`benchmarks/varsr/repo/basicsr/LICENSE`) | Unmodified beyond VARSR's own changes |
| taming-transformers (loss subset) | `benchmarks/taming/repo/taming/` | [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers) | MIT (`benchmarks/taming/repo/License.txt`) | [TAMING_MODIFICATIONS.md](benchmarks/taming/TAMING_MODIFICATIONS.md) |

To keep the repository small and avoid redistributing third-party image datasets, the upstream
example/test images bundled with SwinIR (`testsets/`, `figs/`), SR3 (`dataset/`), and VARSR
(`dataloader/meta_info_AVADataset.csv`) were removed from the vendored copies. They are not needed to
train or evaluate on LiDAR CHMs; some upstream README links inside those folders may therefore be broken.

## Code adapted into GeM-SR's own source tree

| Files | Upstream | License | Notes |
|---|---|---|---|
| `src/utils/logger.py`, `src/utils/lr_schedulers.py`, `src/utils/meter.py` | [markweberdev/maskbit](https://github.com/markweberdev/maskbit) (`utils/`) | Apache-2.0 | Adapted for GeM-SR (each file's header notes the origin). MaskBit in turn credits [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2) and [huggingface/open-muse](https://github.com/huggingface/open-muse) as inspiration. |

A copy of the Apache License 2.0 text is included at `benchmarks/swinir/repo/LICENSE`.

## Data and assets fetched at runtime (not redistributed)

- **LiDAR HHDC cubes** are downloaded from the Hugging Face Hub dataset
  [`anfera236/HHDC`](https://huggingface.co/datasets/anfera236/HHDC) by `jobs/extract_chm.py`.
- **Meta CHMv2 tile** used by `comparison_meta.py` is fetched from Meta's public AWS bucket at runtime and
  is subject to its own terms of use.
- **Pretrained VGG weights** for LPIPS are downloaded by the `lpips` / `torchvision` packages on first use; the
  taming-transformers LPIPS used by `benchmarks/varsr/train_vqvae.py` also downloads its own `vgg.pth`.

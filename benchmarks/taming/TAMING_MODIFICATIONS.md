# taming-transformers (vendored subset)

`repo/taming/` is a minimal subset of [CompVis/taming-transformers](https://github.com/CompVis/taming-transformers)
(MIT, see `repo/License.txt`). It is used only by the VARSR stage-1 tokenizer trainer
(`benchmarks/varsr/train_vqvae.py`) for its **LPIPS + PatchGAN** reconstruction loss,
`VQLPIPSWithDiscriminator`.

Only the files that loss depends on are included:

- `taming/modules/losses/vqperceptual.py` — `VQLPIPSWithDiscriminator`
- `taming/modules/losses/lpips.py` — LPIPS (VGG16 features + learned linear layers)
- `taming/modules/discriminator/model.py` — `NLayerDiscriminator`
- `taming/modules/util.py`, `taming/util.py` — helpers (`ActNorm`, LPIPS weight download)

## Modifications

- `taming/modules/losses/__init__.py` imports `DummyLoss` relatively (`from .vqperceptual import DummyLoss`)
  so the subpackage loads without a conflicting `repo.taming` namespace.
- The LPIPS module expects 3-channel input; rather than editing it, `train_vqvae.py` wraps
  `perceptual_loss.forward` to repeat 1-channel CHMs to 3 channels.

## Runtime downloads

On first use, LPIPS downloads its linear-layer weights (`vgg.pth`, from the URL in `taming/util.py`) and
torchvision downloads ImageNet VGG16 weights.

# About

This project generates new christmas-related images using the technique of generative adversarial networks (GAN).

# Results

# Usage

Requirements are:
* Torch
  * Required packages (most of them should be part of the default torch install, install missing ones with `luarocks install packageName`): `cudnn`, `nn`, `pl`, `paths`, `image`, `optim`, `cutorch`, `cunn`, `cudnn`, `dpnn`, `display`
* Python 2.7 (might work with newer versions, not tested)
  * skimage
  * scipy
  * numpy
* NVIDIA GPU with cudnn3 and 4GB or more memory

To generate the dataset:
* clone the repository and `cd` into it
* `cd dataset`
* `python download_images.py` - This downloads all required images at a rate of 1 image per second, which will take quite some time.
* `python preprocess_images.py` - This will augment and normalize all downloaded images.

To train a network:
* `~/.display/run.js &` - This will start `display`, which is used to plot results in the browser
* Open http://localhost:8000/ in your browser (`display` interface)
* Open a console in the repository directory and use any of the following commands:
  * `th train.lua --profile="baubles32"` - Train a network to generate images of baubles in resolution 32x32
  * `th train.lua --profile="baubles64"` - Train a network to generate images of baubles in resolution 64x64
  * `th train.lua --profile="trees32"` - Train a network to generate images of christmas trees in resolution 32x32
  * `th train.lua --profile="trees64"` - Train a network to generate images of christmas trees in resolution 64x64
  * `th train.lua --profile="snow32"` - Train a network to generate images of snowy landscapes in resolution 32x48
  * `th train.lua --profile="snow64"` - Train a network to generate images of snowy landscapes in resolution 64x96

You can watch how the results of the network improve in the opened browser window. The training will continue until you stop it manually.
You can continue a training at a later time by adding `--network="logs/adversarial.net"` as a parameter.
You can sample images from a trained network using `th sample.lua`.

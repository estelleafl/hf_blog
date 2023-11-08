---
title: "LDM3D - Latent Diffusion Model for 3D"
thumbnail: /blog/assets/bridgetower/thumbnail.png
authors:
- user: sayak
- user: radames
  guest: true
---

# Overview

<!-- {blog_metadata} -->
<!-- {authors} -->

LDM3D is Intel’s Generative AI model that generates image and its depth from text prompts. From it, immersive 360 environments can be created. It is the first-ever custom diffusion model with large-scale training done on Intel AI Supercomputing Cluster powered by Intel Xeon and Habana Gaudi accelerators.
LDM3D leverages Stable Diffusion 1.5 training and generates both image and depth map data from a given text prompt, allowing users to generate RGBD images from text prompts.
 It was finetuned on a dataset constructed from a subset of the LAION-400M dataset, a large-scale image-caption dataset that contains over 400 million image-caption pairs. For more details on the training procedure, you can refer to our [paper](https://arxiv.org/abs/2305.10853)

 We also developed an extension to LDM3D: LDM3D-VR, a suite of diffusion models targeting virtual reality development that includes LDM3D-pano and LDM3D-SR. These models enable the generation of panoramic RGBD based on textual prompts and the upscaling of low-resolution inputs to high-resolution RGBD, respectively. Our models are fine-tuned from existing pretrained models on datasets containing panoramic/high-resolution RGB images, depth maps and captions. You can refer to our [paper](https://arxiv.org/pdf/2311.03226) for more details.


In this blog, we will review these models and how to use some of them on Habana Gaudi and on NVIDIA GPU with [Diffusers](https://github.com/huggingface/diffusers) and [Optimum](https://github.com/huggingface/optimum).

## LDM3D
In order to capture both RGB and depth representations in the latent space, we modified the KL-autoencoder. The KL-encoder now takes as input a RGBD image to create a latent representation of both RGB and depth. After having added noise to this latent, the U-Net will play its role of iteratively denoising it.
A frozen CLIP-text model is used to encode the text prompt and the embedding created from it is then integrated into the U-Net through cross-attention
Finally, the modified KL-decoder decodes the denoised latent representation back to the pixel space as a RGBD output = (RGB, 16-bit grayscale depth map)

<div class="flex justify-center">
    <img src="https://huggingface.co/Intel/ldm3d/resolve/main/model_overview.png">
</div>

## LDM3D-VR

## Applications

-  A HuggingFace Space has been created to visualize panoramic images in a [360 view application](https://huggingface.co/spaces/Intel/ldm3d) using the LDM3D-pano checkpoint.

- We developed an application called DepthFusion, which uses the image-to-image pipeline of LDM3D to create immersive and interactive 360-degree-view experiences using TouchDesigner. This technology has the potential to transform a wide range of industries, from entertainment and gaming to architecture and design.
 Here are [some examples](https://www.youtube.com/watch?v=6oS7gSQzFCI) of use of this application.
[This QR code](assets/model_overview.png) leads to an immersive experience demonstration. If accessed by computer, try and move the mouse around to get a 3D view. If accessed by phone, move around the space with the phone to experience the VR even more.


## How to use

Using the [diffusers library](https://github.com/huggingface/diffusers) from Hugging Face you can easily use our model on NVIDIA GPUS and on CPUs with just a few lines of code. In this section we detail how:


```bash
from diffusers import StableDiffusionLDM3DPipeline
model_name = "Intel/ldm3d-4c"
pipe = StableDiffusionLDM3DPipeline.from_pretrained(model_name)
pipe.to("cuda")
prompt = "A picture of some lemons on a table"
output = pipe( prompt,  width=1024, height=512, guidance_scale=7.0, num_inference_steps=50 )
output.rgb[0].save("lemon_ldm3d_rgb.jpg")
output.depth[0].save("lemon_ldm3d_depth.png")
```
In this example above, you can switch to ldm3d and ldm3d-pano by just updating the model name

```bash
from PIL import Image
from diffusers import StableDiffusionUpscaleLDM3DPipeline

#Upscale the rgb and depth to a resolution of (1024, 1024)
pipe_ldm3d_upscale = StableDiffusionUpscaleLDM3DPipeline.from_pretrained("Intel/ldm3d-sr")
pipe_ldm3d_upscale.to("cuda")

low_res_img = Image.open(f"lemons_ldm3d_rgb.jpg").convert("RGB")
low_res_depth = Image.open(f"lemons_ldm3d_depth.png").convert("L")
outputs = pipe_ldm3d_upscale(prompt="high quality high resolution uhd 4k image", rgb=low_res_img, depth=low_res_depth, num_inference_steps=50, target_res=[1024, 1024])

upscaled_rgb, upscaled_depth =outputs.rgb[0], outputs.depth[0]
upscaled_rgb.save(f"upscaled_lemons_rgb.png")
upscaled_depth.save(f"upscaled_lemons_depth.png")
```

LDM3D is also implemented in the [optimum-habana library](https://github.com/huggingface/optimum-habana) from Hugging Face to use it with Habana Gaudi accelerators. Such implementation makes it really simple to use on a HPU device with a simple “import” from that library. You will be able to benefit from Gaudi’s performances and run LDM3D out of the box.
[After installing the package](https://github.com/huggingface/optimum-habana#install), here is how to run inference:

```bash
from optimum.habana.diffusers import GaudiDDIMScheduler, GaudiStableDiffusionLDM3DPipeline
from optimum.habana.utils import set _seed
model_name = "Intel/ldm3d-4c"
prompt = "A picture of some lemons on a table"
scheduler = GaudiDDIMScheduler.from_pretrained(model_name, subfolder="scheduler")
pipe = GaudiStableDiffusionLDM3DPipeline.from_pretrained( model_name,
 	scheduler=scheduler,
	use_habana=True,
 	use_hpu_graphs=True,
 	gaudi_config="Habana/stable-diffusion")
output = pipe(prompt=["High quality photo of an astronaut riding a horse in space"],
 	num_images_per_prompt=1,
 	batch_size=1,
 	output_type="pil",
	negative_prompt=None)
output.rgb[0].save("lemon_ldm3d_rgb.jpg")
output.depth[0].save("lemon_ldm3d_depth.png")
```


To date, we have released 2 LDM3D checkpoints on Hugging Face:
-        https://huggingface.co/Intel/ldm3d: original checkpoint used to report the results in our paper
Example given the prompt: "A picture of some lemons on a table"
<div class="flex justify-center">
    <img src="https://huggingface.co/Intel/ldm3d/resolve/main/ldm3d_results.png">
</div>

-        https://huggingface.co/Intel/ldm3d-4c: updated checkpoint using a 4-channels packaging. This checkpoint gives improved results
Example given the prompt: "A picture of some lemons on a table"
<div class="flex justify-center">
    <img src="https://huggingface.co/Intel/ldm3d-4c/resolve/main/ldm3d_4c_results.png">
</div>

We also released 2 checkpoints related to LDM3D-VR:
-        https://huggingface.co/Intel/ldm3d-pano: a checkpoint that was finetuned on panoramic images
Example given the prompt: "360 view of a large bedroom"
<div class="flex justify-center">
    <img src="https://huggingface.co/Intel/ldm3d-pano/resolve/main/ldm3d_pano_results.png">
</div>
-        https://huggingface.co/Intel/ldm3d-sr: a checkpoint that upscales rgb and depth images.
<img src="assets/ldm3d_sr_results.png" alt="ldm3d-sr outputs" title="LDM3D-sr outputs">


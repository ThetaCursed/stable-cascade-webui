import os
import random
import string
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import gradio as gr
import gc
from PIL import Image


def generate_images(prompt, negative_prompt, num_images_per_prompt, height, width, mseed, guidance_scale, prior_inference_steps, decoder_inference_steps=10):

    device = "cuda"

    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to("cuda")
    prior.enable_xformers_memory_efficient_attention()

    prior.safety_checker = None
    prior.requires_safety_checker = False

    mseed = int(mseed)

    outputs = []
    filenames = []  # Store filenames for captions
    for _ in range(num_images_per_prompt):
        prior_output = prior(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            mseed=mseed,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,  # Generate one image at a time
            num_inference_steps=prior_inference_steps,
            generator=torch.Generator(device=device).manual_seed(mseed)
        )

        outputs.append(prior_output)
        # Generate unique filename for each image
        filename = ''.join(random.choices(string.ascii_lowercase, k=10)) + ".png"  # Save as PNG
        filenames.append(filename)

        mseed += 1  # Increment mseed

    del prior
    gc.collect()
    torch.cuda.empty_cache()

    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=torch.float16).to("cuda")
    decoder.enable_xformers_memory_efficient_attention()

    decoder.safety_checker = None
    decoder.requires_safety_checker = False

    decoded_images = []
    for prior_output, filename in zip(outputs, filenames):
        decoder_output = decoder(
            image_embeddings=prior_output.image_embeddings.half(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,  # Guidance scale typically set to 0 for decoder as guidance is applied in the prior
            output_type="pil",
            num_inference_steps=decoder_inference_steps
        ).images
        decoded_images.extend(zip(decoder_output, [filename] * len(decoder_output)))  # Zip images with their filenames

    del decoder
    gc.collect()
    torch.cuda.empty_cache()

    return decoded_images


def save_images(decoded_images, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    for img, filename in decoded_images:
        path = os.path.join(output_dir, filename)
        img.save(path)
        saved_paths.append(path)
    return saved_paths


def web_demo():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                text2image_prompt = gr.Textbox(
                    lines=1,
                    placeholder="Prompt",
                    show_label=False,
                )

                text2image_negative_prompt = gr.Textbox(
                    lines=1,
                    placeholder="Negative Prompt",
                    show_label=False,
                    value="ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers"
                )
                with gr.Row():
                    with gr.Column():
                        text2image_num_images_per_prompt = gr.Slider(
                            minimum=1,
                            maximum=64,
                            value=1,
                            step=1,
                            label="Number of images",
                        )

                        text2image_height = gr.Slider(
                            minimum=1024,
                            maximum=4096,
                            step=64,
                            value=1024,
                            label="Image Height",
                        )

                        text2image_width = gr.Slider(
                            minimum=1024,
                            maximum=4096,
                            step=64,
                            value=1024,
                            label="Image Width",
                        )

                        text2image_seed = gr.Number(
                            minimum=1,
                            maximum=4000000000,
                            value=1,
                            label="Seed",
                        )
                        with gr.Row():
                            with gr.Column():
                                text2image_guidance_scale = gr.Slider(
                                    minimum=0.5,
                                    maximum=15,
                                    step=0.1,
                                    value=4.0,
                                    label="Guidance Scale",
                                )
                                text2image_prior_inference_step = gr.Slider(
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    value=20,
                                    label="Prior Inference Step",
                                )

                                text2image_decoder_inference_step = gr.Slider(
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    value=10,
                                    label="Decoder Inference Step",
                                )
                text2image_predict = gr.Button(value="Generate Image")

            with gr.Column():
                output_image = gr.Gallery(
                    label="Generated images",
                    show_label=False,
                    elem_id="gallery",
                )

            text2image_predict.click(
                fn=lambda prompt, negative_prompt, num_images_per_prompt, height, width, mseed, guidance_scale,
                        prior_inference_steps, decoder_inference_steps: save_images(
                    generate_images(prompt, negative_prompt, num_images_per_prompt, height, width, mseed,
                                     guidance_scale, prior_inference_steps, decoder_inference_steps)
                ),
                inputs=[
                    text2image_prompt,
                    text2image_negative_prompt,
                    text2image_num_images_per_prompt,
                    text2image_height,
                    text2image_width,
                    text2image_seed,
                    text2image_guidance_scale,
                    text2image_prior_inference_step,
                    text2image_decoder_inference_step
                ],
                outputs=[output_image],
            )

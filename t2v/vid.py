import os
import inspect
import fire
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler
import numpy as np
import torch
from torch import autocast
from torchvision.utils import make_grid
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageDraw, ImageFont
    
@torch.no_grad()
#from stable diffusion main code
def diffuse(
        pipe,
        cond_embeddings,  # text conditioning, should be (1, 77, 768)
        cond_latents,  # image conditioning, should be (1, 4, 64, 64)
        uncond_embed,
        num_inference_steps,
        guidance_scale,
        eta,
        timestep=1,
        flag=1,
):
    torch_device = cond_latents.get_device()
    #TODO: Reduce size of embeddings

    text_embeddings = torch.cat([uncond_embed, cond_embeddings])

    if isinstance(cond_latents, (torch.Tensor, Image.Image, list)):
        cond_latents = cond_latents.to(device=torch_device)

    # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
    if isinstance(pipe.scheduler, LMSDiscreteScheduler) or isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
        cond_latents = cond_latents * pipe.scheduler.sigmas[0]

    # init the scheduler
    accepts_offset = "offset" in set(inspect.signature(pipe.scheduler.set_timesteps).parameters.keys())
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs["offset"] = 1
    pipe.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

    accepts_eta = "eta" in set(inspect.signature(pipe.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # diffuse!
    for i, t in enumerate(pipe.scheduler.timesteps):

        latent_model_input = torch.cat([cond_latents] * 2)
        if isinstance(pipe.scheduler, LMSDiscreteScheduler) or isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
            sigma = pipe.scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)

        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # cfg
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if isinstance(pipe.scheduler, LMSDiscreteScheduler) or isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
            cond_latents = pipe.scheduler.step(noise_pred, i+1, cond_latents, **extra_step_kwargs)["prev_sample"]
        else:
            cond_latents = pipe.scheduler.step(noise_pred, t, cond_latents, **extra_step_kwargs)["prev_sample"]

    # scale and decode the image latents with vae
    cond_latents = 1 / 0.18215 * cond_latents
    image = pipe.vae.decode(cond_latents)

    # generate output numpy image as uint8
    image = (image['sample']/ 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    if flag == 0:
        image = (image[0] * 255 * timestep).astype(np.uint8)
    elif flag == 1:
        image = (image[0] * 255).astype(np.uint8)
    elif flag == -1:
        image = (image[0] * 255 * (1-timestep)).astype(np.uint8)


    return image


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 """

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def lerp(t, v0, v1, flag, t_threshold = 0.8):
    """ helper function to spherically interpolate two arrays v1 v2 """

    if not isinstance(v1, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))

    if flag == False:
        v2 = v0
    else:
        s1 = (t - t_threshold)/(1 - t_threshold)
        s0 = 1 - s1
        v2 = s0*v0 + s1*v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def new_embeds(embed1, embed2, chunk_size=8):
    device = embed1.device
    embed1_a, *embed1 = embed1
    embed2_a, *embed2 = embed2

    final_embed_cuda = []

    for i in range(int(len(embed1_a) / chunk_size) + 1):
        subembed_a = embed1_a[i * chunk_size:i * chunk_size + chunk_size]
        new_size = len(subembed_a)
        ref = []
        sim = []

        for j in range(0, len(embed2_a) - (new_size - 1), 1):
            subembed_b = embed2_a[j:j + new_size, :]
            similarity = cosine_similarity(subembed_a.cpu().flatten().reshape(1, -1), subembed_b.cpu().flatten().reshape(1, -1))
            sim.append(similarity[0][0])
            ref.append(subembed_b)

        most_similar_index = np.argmax(sim)
        most_similar_array = ref[most_similar_index]
        final_embed_cuda.append(most_similar_array)

    final_embed = [tensor.cpu() for tensor in final_embed_cuda]
    embed2_final = np.concatenate(final_embed, axis=0)

    return torch.from_numpy(np.expand_dims(embed2_final, axis=0)).to(device)


def most_similar(chunk2, embed1):
    device = embed1.device
    # embed1_a, *embed1 = embed1

    ref = []
    sim = []
    for j in range(0, len(embed1), 1):
        subembed_b = embed1[j:j + 1, :]
        similarity = cosine_similarity(chunk2.cpu().flatten().reshape(1, -1), subembed_b.cpu().flatten().reshape(1, -1))
        sim.append(similarity[0][0])
        ref.append(subembed_b)

    most_similar_index = np.argmax(sim)
    most_similar_array = ref[most_similar_index]
    embed1_mod = ref[:most_similar_index] + ref[most_similar_index + 1:]
    final_embed = [tensor.cpu() for tensor in embed1_mod]
    embed1_final = np.concatenate(final_embed, axis=0)

    return most_similar_index, most_similar_array, torch.from_numpy(embed1_final).reshape(embed1_final.shape[1]).to(device)


def create_video(
        prompts=["man day phoned someone",
                 "her continent stands running to dice someone",
                 "you am need her brightest toolbox middle her house"],  # prompts to dream about

        seeds=[100, 200, 300],
        gpu=1,  # id of the gpu to run on
        chunk_interpolation = False,
        rootdir='./results',
        num_steps=100,  # number of steps between each pair of sampled points
        frame_index = 0,
        # --------------------------------------
        # args you probably don't want to change
        num_inference_steps=100,
        guidance_scale=17.5,
        eta=0.1,
        width=512,
        height=512,
        # --------------------------------------
):
    assert len(prompts) == len(seeds)
    assert torch.cuda.is_available()
    assert height % 8 == 0 and width % 8 == 0

    embeds = []
    ims = []

    # init the output dir
    # outdir = os.path.join(rootdir, name)
    outdir = rootdir
    os.makedirs(outdir, exist_ok=True)

    # # init all of the models and move them to a given GPU
    pipe = StableDiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0")

    torch_device = f"cuda:{gpu}"
    # pipe.safety_checker = False
    pipe.unet.to(torch_device)
    pipe.vae.to(torch_device)
    pipe.text_encoder.to(torch_device)

    negative_prompts = 'text, blurry, morbid, longbody, lowres, bad anatomy, bad hands, missing fingers, low quality, deformed body, ugly, unrealistic, nude, naked'

    uncond_input = pipe.tokenizer(negative_prompts, padding='max_length', max_length=60, truncation=False, return_tensors="pt")
    with torch.no_grad():
        uncond_embed = pipe.text_encoder(uncond_input.input_ids.to(torch_device))[0]
    # get the conditional text embeddings based on the prompts
    prompt_embeddings = []
    for prompt in prompts:
        # keyword_prompt = kw_model.extract_keywords(prompt, keyphrase_ngram_range=(1, 7))[0][0]
        text_input = pipe.tokenizer(
            # keyword_prompt,
            prompt,
            padding='max_length',
            max_length=uncond_embed.shape[1],
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            embed = pipe.text_encoder(text_input.input_ids.to(torch_device))[0]

        prompt_embeddings.append(embed)

    # Take first embed and set it as starting point, leaving rest as list we'll loop over.
    prompt_embedding_a, *prompt_embeddings = prompt_embeddings
    # Take first seed and use it to generate init noise
    init_seed, *seeds = seeds
    init_a = torch.randn(
        (1, pipe.unet.in_channels, height // 8, width // 8),
        device=torch_device,
        generator=torch.Generator(device=torch_device).manual_seed(init_seed.item())
    )

    for p, prompt_embedding_b in enumerate(prompt_embeddings):

        init_b = torch.randn(
            (1, pipe.unet.in_channels, height // 8, width // 8),
            generator=torch.Generator(device=torch_device).manual_seed(seeds[p].item()),
            device=torch_device
        )
        print(prompt_embedding_b.shape)

        for i, t in enumerate(np.linspace(0, 1, num_steps)):

            print("Generating", frame_index)
            if chunk_interpolation:
                prompt_embedding_b_mod = new_embeds(prompt_embedding_a, prompt_embedding_b)
                cond_embedding = slerp(float(t), prompt_embedding_a, prompt_embedding_b_mod)
            else:
                cond_embedding = slerp(float(t), prompt_embedding_a, prompt_embedding_b)

            init = slerp(float(t), init_a, init_b)

            with autocast("cuda"):
                image = diffuse(pipe, cond_embedding, init, uncond_embed, num_inference_steps, guidance_scale, eta, 1)

            im = Image.fromarray(image)
            # outpath = os.path.join(outdir, 'frame%06d.jpg' % 0)
            outpath = os.path.join(outdir, 'frame%06d.jpg' % frame_index)
            im.save(outpath)
            frame_index += 1
            embeds.append(cond_embedding)
            ims.append(im)


        if chunk_interpolation:
            prompt_embedding_a = prompt_embedding_b_mod
        else:
            prompt_embedding_a = prompt_embedding_b

        init_a = init_b

    return frame_index, embeds, ims
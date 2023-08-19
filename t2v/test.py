from t2v.vid import *

def img2emb(pipe, image, seed, device):
    # transform = transforms.Compose([
    #     transforms.PILToTensor()
    # ])
    generator = torch.Generator(device=device).manual_seed(seed)
    # image = transform(image).to(device, dtype=torch.float32)
    if isinstance(image, Image.Image):
        image = [image]

    if isinstance(image[0], Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample= Image.Resampling.BICUBIC))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image).to(device)

    image2 = Image.fromarray(image.cpu().numpy())
    image2.save('1.jpg')
    init_latents = pipe.vae.encode(image).latent_dist.sample()
    init_latents = pipe.scheduler.init_noise_sigma *init_latents
    return init_latents.to(device)

prompt1 = "A man walking on the street"
prompt2 = "A man swimming in the ocean"
negative_prompt = 'nude, naked, text, digits, worst quality, blurry, morbid, poorly drawn face, bad anatomy,distorted face, disfiguired, bad hands, missing fingers,cropped, deformed body, bloated, ugly, unrealistic'


pipe = StableDiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0")

torch_device = "cuda"
# pipe.safety_checker = False
pipe.unet.to(torch_device)
pipe.vae.to(torch_device)
pipe.text_encoder.to(torch_device)

prompt1_tok = pipe.tokenizer(prompt1, padding='max_length', max_length=60, truncation=True,
                                return_tensors="pt")
with torch.no_grad():
    prompt1_embed = pipe.text_encoder(prompt1_tok.input_ids.to(torch_device))[0]

prompt2_tok = pipe.tokenizer(prompt2, padding='max_length', max_length=60, truncation=True,
                                return_tensors="pt")
with torch.no_grad():
    prompt2_embed = pipe.text_encoder(prompt2_tok.input_ids.to(torch_device))[0]

uncond_input = pipe.tokenizer(negative_prompt, padding='max_length', max_length=60, truncation=True, return_tensors="pt")
with torch.no_grad():
    uncond_embed = pipe.text_encoder(uncond_input.input_ids.to(torch_device))[0]

frame_index = 0
img_size = 512
init_image_1 = pipe(prompt1, height = img_size, width = img_size).images[0]
init_image_2 = pipe(prompt2, height = img_size, width = img_size).images[0]

init_a = img2emb(pipe, init_image_1, 1024, torch_device)
# init_b = img2emb(pipe, init_image_2, 1024, torch_device)
init_b = torch.randn(
        (1, pipe.unet.in_channels, img_size // 8, img_size // 8),
        generator=torch.Generator(device=torch_device).manual_seed(1024),
        device=torch_device
    )

for i, t in enumerate(np.linspace(0, 1,20)):
    print("generating... ", frame_index)
    cond_embedding = slerp(float(t), prompt1_embed, prompt2_embed)
    init = slerp(float(t), init_a.detach(), init_b.detach())

    with autocast("cuda"):
        im = diffuse(pipe, cond_embedding, init, uncond_embed, 50, 15, 0.05)

    # image = Image.fromarray(im).convert('RGB')
    image = Image.fromarray(im)
    outpath = os.path.join('./t2v/results/', 'frame%06d.jpg' % frame_index)
    image.save(outpath)
    frame_index += 1
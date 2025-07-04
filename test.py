import argparse, os, sys, glob
import PIL
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything
sys.path.append(os.path.dirname(sys.path[0]))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from transformers import CLIPProcessor, CLIPModel

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((512, 512), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
        
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std       

def add_footer(image, text, font_size=18):
    w, h = image.size
    footer_height = font_size + 8
    new_image = Image.new("RGB", (w, h + footer_height), (255, 255, 255))
    new_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_image)
    try: 
        font =  ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    text_w, text_h = draw.textsize(text, font=font)
    draw.text(((w - text_w) // 2, h + 4), text, fill=(0, 0, 0), font=font)
    return new_image

def infer(sampler, model, prompt = '', content_dir = '', style_dir='',ddim_steps = 50,strength = 0.5, seed=42, device='cuda'):
    ddim_eta=0.0
    n_iter=1
    C=4
    f=8
    n_samples=1
    n_rows=0
    scale=10.0
    
    precision="autocast"
    outdir="outputs"
    seed_everything(seed)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    data = [batch_size * [prompt]]
    
    style_image = load_img(style_dir)
    style_image = repeat(style_image, '1 ... -> b ...', b=batch_size)
    style_image = style_image.to(device)
    style_latent = model.get_first_stage_encoding(model.encode_first_stage(style_image))  # move to latent space

    content_name =  content_dir.split('/')[-1].split('.')[0]
    content_image = load_img(content_dir)
    content_image = repeat(content_image, '1 ... -> b ...', b=batch_size)
    content_image = content_image.to(device)
    content_latent = model.get_first_stage_encoding(model.encode_first_stage(content_image))  # move to latent space

    init_latent = adain(content_latent,style_latent) # with AdaIN 
    # init_latent = content_latent # without AdaIN 

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope('cpu'):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                # test = True
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if scale != 1.0:
                            # if test:
                                # from IPython import embed
                                # embed()
                                # test = False
                            uc = model.get_learned_conditioning(batch_size * [""], style_image)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        
                        c= model.get_learned_conditioning(prompts, style_image)

                        # img2img
                        # stochastic inversion
                        t_enc = int(strength * 1000) 
                        x_noisy = model.q_sample(x_start=init_latent, t=torch.tensor([t_enc]*batch_size).to(device))
                        model_output = model.apply_model(x_noisy, torch.tensor([t_enc]*batch_size).to(device), c)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device),\
                                                          noise = model_output, use_original_steps = True)
            
                        t_enc = int(strength * ddim_steps)
                        samples = sampler.decode(z_enc, c, t_enc, 
                                                unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,)
                        print(z_enc.shape, uc.shape, t_enc)

                        # txt2img
            #             noise  =torch.randn_like(content_latent)
            #             samples, intermediates =sampler.sample(ddim_steps,1,(4,512,512),c,verbose=False, eta=1.,x_T = noise,
            #    unconditional_guidance_scale=scale,
            #    unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)

                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            # base_count += 1
                        all_samples.append(x_samples)

                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                output = Image.fromarray(grid.astype(np.uint8))
                name = content_name+'-'+prompt+'-'f'{seed}-{strength}-{ddim_steps}'+'.png'
                # output.save(os.path.join(outpath, name))

                toc = time.time()
    return output

if __name__ == "__main__":
    style_dir = 'data/inst_train/4593097_1.jpg'
    content_dir = '/mnt/md0/projects/ai_headshot/style_transfer_data/content-image/woman/w1.png'
    embed_dir = 'logs/4593097_1.jpg2025-07-02-04-26-41-_v1-finetune/testtube/version_0/checkpoints/embeddings_gs-6099.pt'
    config="configs/stable-diffusion/v1-inference.yaml"
    ckpt="models/sd/sd-v1-4.ckpt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load(f"{config}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir", type=str, default=content_dir, help="Path to content image")
    parser.add_argument("--style_dir", type=str, default=style_dir, help="Path to style image")
    parser.add_argument("--embed_dir", type=str, default=embed_dir, help="Path to style embedding")
    args = parser.parse_args()
    
    import re
    style_id = re.search(r'/([^/]+)\.(?:jpg|jpeg|png)$', args.style_dir, re.IGNORECASE).group(1)

    model = load_model_from_config(config, f"{ckpt}", device)
    sampler = DDIMSampler(model)
    model.embedding_manager.load(args.embed_dir)

    model = model.to(device)
    if hasattr(model, "cond_stage_model"):
        model.cond_stage_model = model.cond_stage_model.to(device)
        if hasattr(model.cond_stage_model, "image_encoder"):
            model.cond_stage_model.image_encoder = model.cond_stage_model.image_encoder.to(device)
    
    model.embedding_manager = model.embedding_manager.to(device)

    ddim_steps_list = [50, 150]
    strength_list = [0.6, 0.8]
    seed_list = [42, 1234]
    
    prompts_list = ['a woman, masterpiece, best quality, high quality'] # 'a wedding couple, masterpiece, best quality, high quality'
    header_titles = ["Content", "Style"] + \
                [f"ddim_steps={d}" for d in ddim_steps_list] + \
                [f"seed={s}" for s in seed_list] + \
                [f"strength={st}" for st in strength_list]

    img_size = 512
    content_img_pil = Image.open(args.content_dir).convert("RGB").resize((img_size, img_size), resample=PIL.Image.LANCZOS)
    style_img_pil = Image.open(args.style_dir).convert("RGB").resize((img_size, img_size), resample=PIL.Image.LANCZOS)
    content_img_tensor = torch.from_numpy(np.array(content_img_pil)).permute(2, 0, 1).float() / 255.0
    style_img_tensor = torch.from_numpy(np.array(style_img_pil)).permute(2, 0, 1).float() / 255.0

    grid_rows = []
    for prompt in prompts_list:
        row_images = []
        # First column: content image
        row_images.append(content_img_tensor)
        # Second column: style image
        row_images.append(style_img_tensor)
        # Next columns: ddim_steps
        for ddim_steps in ddim_steps_list:
            output = infer(
                sampler=sampler,
                prompt=prompt,
                content_dir=args.content_dir,
                style_dir=args.style_dir,
                ddim_steps=ddim_steps,
                strength=strength_list[0],
                seed=seed_list[0],
                model=model,
                device=device
            )
            img_tensor = torch.from_numpy(np.array(output)).permute(2, 0, 1).float() / 255.0
            row_images.append(img_tensor)
        # Next columns: seed
        for seed in seed_list:
            output = infer(
                sampler=sampler,
                prompt=prompt,
                content_dir=args.content_dir,
                style_dir=args.style_dir,
                ddim_steps=ddim_steps_list[0],
                strength=strength_list[0],
                seed=seed,
                model=model,
                device=device
            )
            img_tensor = torch.from_numpy(np.array(output)).permute(2, 0, 1).float() / 255.0
            row_images.append(img_tensor)
        # Next columns: strength
        for strength in strength_list:
            output = infer(
                sampler=sampler,
                prompt=prompt,
                content_dir=args.content_dir,
                style_dir=args.style_dir,
                ddim_steps=ddim_steps_list[0],
                strength=strength,
                seed=seed_list[0],
                model=model,
                device=device
            )
            img_tensor = torch.from_numpy(np.array(output)).permute(2, 0, 1).float() / 255.0
            row_images.append(img_tensor)
        # Stack this row horizontally
        row_tensor = torch.stack(row_images)
        row_grid = make_grid(row_tensor, nrow=len(row_images))
        grid_rows.append(row_grid)

    # Create header images for each column
    img_w, img_h = img_size, img_size
    header_height = 40
    header_images = []
    for title in header_titles:
        header_img = Image.new("RGB", (img_w, header_height), (255, 255, 255))
        draw = ImageDraw.Draw(header_img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        text_w, text_h = draw.textsize(title, font=font)
        draw.text(((img_w - text_w) // 2, (header_height - text_h) // 2), title, fill=(0, 0, 0), font=font)
        header_images.append(header_img)
    header_tensors = [torch.from_numpy(np.array(h)).permute(2, 0, 1).float() / 255.0 for h in header_images]
    header_row = make_grid(torch.stack(header_tensors), nrow=len(header_tensors))

    # Stack header row and image grid
    final_grid = torch.cat([header_row, *grid_rows], dim=1)

    # Convert to image and save
    final_grid_img = (final_grid * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    final_grid_pil = Image.fromarray(final_grid_img)
    name = f"outputs/{style_id}_grid.png"
    final_grid_pil.save(name)
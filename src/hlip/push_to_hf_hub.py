#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath('..'))

import json
import re
import shutil
import argparse
import textwrap
from pathlib import Path
import importlib.util

import torch
import safetensors.torch

# HF Hub
from huggingface_hub import create_repo, upload_folder

# OpenCLIP
from open_clip.factory import _MODEL_CONFIGS
from open_clip import create_model_and_transforms, get_tokenizer
from open_clip.push_to_hf_hub import save_for_hf

# Ensure your custom code is importable locally
import hlip
import hlip.visual_encoder
import hlip.visual_encoder_rope


def load_checkpoint(path: str):
    """Robust torch.load that uses weights_only when available."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def strip_module_prefix(sd: dict):
    """Remove 'module.' prefix from DDP state dict keys."""
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def find_package_paths(pkg_name: str):
    """
    Return all source directories for a package, including namespace packages.
    We will merge-copy them into the staging dir.
    """
    paths = []
    spec = importlib.util.find_spec(pkg_name)
    if spec and spec.submodule_search_locations:
        paths.extend([Path(p).resolve() for p in spec.submodule_search_locations])

    # Fallback scan of sys.path for a folder named pkg_name
    if not paths:
        for p in map(Path, sys.path):
            cand = (p / pkg_name)
            if cand.is_dir():
                paths.append(cand.resolve())

    if not paths:
        raise RuntimeError(f"Could not locate package '{pkg_name}' on disk.")
    # Deduplicate
    uniq = []
    seen = set()
    for p in paths:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def vendor_namespace_package(pkg_name: str, dest_dir: Path):
    """Copy a (possibly namespace) package into dest_dir/pkg_name."""
    src_paths = find_package_paths(pkg_name)
    pkg_dst = dest_dir / pkg_name
    pkg_dst.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.so", ".git", ".pytest_cache")
    for src in src_paths:
        shutil.copytree(src, pkg_dst, dirs_exist_ok=True, ignore=ignore)
    # Convert to regular package in the snapshot for simpler downstream imports
    (pkg_dst / "__init__.py").touch(exist_ok=True)
    return pkg_dst


def infer_publish_date(explicit_date: str | None, repo_id: str, ckpt_path: Path) -> str:
    """Return YYYY-MM-DD for HF metadata config.json."""
    if explicit_date:
        return explicit_date

    for candidate in (repo_id, str(ckpt_path)):
        m = re.search(r"(20\d{2}-\d{2}-\d{2})", candidate)
        if m:
            return m.group(1)

    raise ValueError(
        "Could not infer date. Pass --date YYYY-MM-DD (e.g. --date 2025-10-08)."
    )


def build_readme(repo_id: str, model_name: str, example_study_dir_name: str) -> str:
    """Generate a HF README with a runnable zero-shot loading example."""
    template = textwrap.dedent(
        """\
        ---
        library_name: open_clip
        tags:
          - open_clip
          - medical-imaging
          - zero-shot-image-classification
        ---

        # HLIP Zero-Shot Model

        This repository contains a custom HLIP/OpenCLIP checkpoint and the vendored `hlip` package required to
        register the custom visual encoder for loading.

        ## Example: zero-shot inference from Hugging Face Hub

        ```python
        from pathlib import Path
        import os, sys, json, torch, importlib

        from huggingface_hub import snapshot_download
        from open_clip.factory import _MODEL_CONFIGS
        from open_clip import create_model_and_transforms, get_tokenizer, build_zero_shot_classifier

        import safetensors.torch as st
        from torchvision.transforms import Normalize
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


        def loader(study_path: str, num_slices: int):
            \"\"\"
            study_path: folder containing per-slice tensors saved with torch.save()
                        each file is a [C, H, W] or [C, H, W, 1] tensor in [0, 255]
            returns:   image tensor of shape [1, n_scans, 1, D, H, W]
            \"\"\"
            imgs = []
            for scan in [os.path.join(study_path, p) for p in os.listdir(study_path)]:
                # load image tensor
                img = torch.load(scan, weights_only=True)
                if len(img.shape) == 4:
                    # [C, H, W, 1] -> [C, H, W]
                    img = img[:, :, :, 0]
                img = img.float() / 255.0  # [C, H, W]
                _, h, w = img.shape

                # pad to square
                size = max(h, w)
                pad_h = size - h
                pad_w = size - w
                left = pad_w // 2
                right = pad_w - left
                top = pad_h // 2
                bottom = pad_h - top
                img = torch.nn.functional.pad(
                    img, (left, right, top, bottom), mode="constant", value=0
                )

                # resize to 256, make depth=num_slices, center-crop to 224
                img = torch.nn.functional.interpolate(
                    img[None, ...], size=(256, 256), mode="bilinear"
                )[0]
                img = torch.nn.functional.interpolate(
                    img[None, None, ...], size=(num_slices, 256, 256), mode="nearest-exact"
                )[0, 0]
                img = img[:, 16:240, 16:240]  # [D, 224, 224]

                # normalize (scalar mean/std across slices-as-channels)
                normalizer = Normalize(
                    torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(),
                    torch.as_tensor(IMAGENET_DEFAULT_STD).mean(),
                )
                img = normalizer(img[None, ...])  # [1, D, H, W]
                imgs.append(img)

            # [1, n_scans, 1, D, H, W]
            return torch.stack(imgs, dim=0)[None, ...]


        # ---- constants ----
        REPO_ID = "__REPO_ID__"
        MODEL_NAME = "__MODEL_NAME__"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # -------------------

        # 1) download snapshot and make vendored package importable
        repo_dir = Path(snapshot_download(repo_id=REPO_ID))
        sys.path.append(str(repo_dir))
        importlib.invalidate_caches()
        print(f"[OK] repo_dir = {repo_dir}")

        # 2) import your registry so timm/OpenCLIP knows the custom visual encoder
        import hlip.visual_encoder  # registers the custom visual encoder with timm
        import hlip.visual_encoder_rope  # optional depending on model; safe to import if vendored

        # 3) load the vendored HLIP model config and register it under MODEL_NAME
        cfg_path = repo_dir / "hlip" / "model_configs" / f"{MODEL_NAME}.json"
        model_cfg = json.loads(cfg_path.read_text())
        model_cfg.setdefault("text_cfg", {})
        model_cfg["text_cfg"]["hf_tokenizer_name"] = REPO_ID
        _MODEL_CONFIGS[MODEL_NAME] = model_cfg
        print("[OK] registered MODEL_CONFIGS key:", MODEL_NAME)

        # 4) build model and tokenizer
        model, _, _ = create_model_and_transforms(
            MODEL_NAME,
            device=DEVICE,
            output_dict=True,
        )
        tokenizer = get_tokenizer(MODEL_NAME)
        print("[OK] model built on", DEVICE)
        print("[OK] tokenizer ready")

        # 5) load pretrained weights from the snapshot (prefer safetensors)
        weight_path = None
        for fname in ("model.safetensors", "pytorch_model.bin"):
            p = repo_dir / fname
            if p.exists():
                weight_path = p
                break
        assert weight_path is not None, "No weights found in repo snapshot."

        if weight_path.suffix == ".safetensors":
            state_dict = st.load_file(str(weight_path))
        else:
            state_dict = torch.load(str(weight_path), map_location="cpu")

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"[OK] loaded weights: {weight_path.name} | "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

        # 6) build zero-shot classifier for brain MRI labels
        from hlip.zeroshot_metadata_pubbrain5 import PROMPTS, TEMPLATES

        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=PROMPTS["prompt"],
            templates=TEMPLATES["template"],
            num_classes_per_batch=None,  # use all classes
            device=DEVICE,
            use_tqdm=False,
        )

        # 7) example data and inference
        # This snapshot includes an example study under docs/.
        # Replace this with your own study folder of per-slice tensors if needed.
        study_dir = repo_dir / "docs" / "__EXAMPLE_STUDY_DIR_NAME__"
        image = loader(str(study_dir), num_slices=48).to(DEVICE, non_blocking=True)

        model.eval()
        with torch.no_grad():
            output = model(image=image)  # image: [1, n_scans, 1, D, H, W]
            # HLIP returns per-scan image features; use the first scan token to match eval scripts.
            image_features = output["image_features"][:, 0, :]  # [1, feature_dim]
            logit_scale = output["logit_scale"]
            logits_per_image = logit_scale * (image_features @ classifier)  # [1, num_classes]
            probs = logits_per_image.softmax(dim=-1).detach().cpu()

        print("Zero-shot class probabilities:")
        for i, prompt in enumerate(PROMPTS["prompt"]):
            print(f"{prompt}: {float(probs[0, i]):.4f}")
        ```
        """
    )
    return (
        template
        .replace("__REPO_ID__", repo_id)
        .replace("__MODEL_NAME__", model_name)
        .replace("__EXAMPLE_STUDY_DIR_NAME__", example_study_dir_name)
    )


def resolve_canonical_repo_id(requested_repo_id: str, create_repo_result) -> str:
    """
    Normalize repo_id to namespace/name when create_repo inferred the namespace.

    huggingface_hub versions differ in return type (RepoUrl/string), so parse defensively.
    """
    if "/" in requested_repo_id:
        return requested_repo_id

    # Newer versions return RepoUrl with .repo_id
    repo_id_attr = getattr(create_repo_result, "repo_id", None)
    if isinstance(repo_id_attr, str) and repo_id_attr:
        return repo_id_attr

    # Fallback: parse returned URL/string
    result_str = str(create_repo_result).strip()
    if "huggingface.co/" in result_str:
        suffix = result_str.split("huggingface.co/", 1)[1].strip("/")
        if "/" in suffix:
            return suffix

    return requested_repo_id


def main():
    parser = argparse.ArgumentParser("Push custom OpenCLIP model + code to HF Hub")
    parser.add_argument("--repo-id", type=str,
                        default="zch0414/hlip-2025_10_08",
                        help="HF Hub repo id like user_or_org/name")
    parser.add_argument("--model-name", type=str,
                        default="ablate_seqposemb_clip_vit_base_multiscan_h2_dinotxt1568",
                        help="Must match your JSON config filename without .json")
    parser.add_argument("--ckpt", type=str, required=False,
                        default="/nfs/turbo/umms-tocho-snr/exp/chuizhao/hlip/ablation/DATE_2025_10_08-13_22_14-MODEL_ablate_seqposemb_clip_vit_base_multiscan_h2_dinotxt1568/checkpoints/epoch_5.pt",
                        help="Path to your trained checkpoint .pt")
    parser.add_argument("--model-config-dir", type=str, default="./model_configs",
                        help="Folder containing your OpenCLIP JSON configs")
    parser.add_argument("--tmpdir", type=str, default="../../hf_staging",
                        help="Local staging directory used before uploading to HF Hub")
    parser.add_argument(
        "--docs-src-dir",
        type=str,
        default="/nfs/turbo/umms-tocho-snr/exp/chuizhao/data/pub_brain_5/brats23/train/adult_glioma/BraTS-GLI-00459-000",
        help="Local directory to copy into HF repo as docs/<basename> for an example study",
    )
    parser.add_argument("--date", type=str, default=None,
                        help="YYYY-MM-DD metadata written to config.json (default: infer from repo-id or ckpt path)")
    parser.add_argument("--private", action="store_true", help="Create/keep repo private")
    parser.add_argument("--commit-message", type=str,
                        default="Add weights (safetensors+bin), config, tokenizer, and custom hlip code.",
                        help="Commit message for the upload")
    args = parser.parse_args()

    repo_id = args.repo_id
    model_name = args.model_name
    ckpt_path = Path(args.ckpt)
    model_cfg_dir = Path(args.model_config_dir)
    tmpdir = Path(args.tmpdir)
    docs_src_dir = Path(args.docs_src_dir)
    publish_date = infer_publish_date(args.date, repo_id, ckpt_path)

    assert model_cfg_dir.is_dir(), f"Missing model config dir: {model_cfg_dir}"
    assert ckpt_path.is_file(), f"Missing checkpoint: {ckpt_path}"
    assert docs_src_dir.is_dir(), f"Missing docs source dir: {docs_src_dir}"

    print(f"[1/7] Ensuring repo exists: {repo_id}")
    repo_create_result = create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
    repo_id = resolve_canonical_repo_id(repo_id, repo_create_result)
    print(f"      Using canonical repo id: {repo_id}")

    print(f"[2/7] Registering model configs from {model_cfg_dir}")
    for p in model_cfg_dir.glob("*.json"):
        with p.open("r") as f:
            _MODEL_CONFIGS[p.stem] = json.load(f)

    print(f"[3/7] Building model: {model_name}")
    model, _, _ = create_model_and_transforms(model_name, device="cpu", output_dict=True)

    print(f"[4/7] Loading checkpoint: {ckpt_path}")
    ckpt = load_checkpoint(str(ckpt_path))
    sd = ckpt.get("state_dict", ckpt)
    sd = strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  Warning: missing keys: {len(missing)} (showing first 5): {missing[:5]}")
    if unexpected:
        print(f"  Warning: unexpected keys: {len(unexpected)} (showing first 5): {unexpected[:5]}")

    print(f"[5/7] Preparing tokenizer and model_config")
    tokenizer = get_tokenizer(model_name)
    with (model_cfg_dir / f"{model_name}.json").open("r") as f:
        model_config = json.load(f)
    model_config.setdefault("text_cfg", {})
    model_config["text_cfg"]["hf_tokenizer_name"] = repo_id

    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)
    print(f"[6/7] Staging to {tmpdir}")

    # 6a) Write HF config + tokenizer, but skip weights; we will save contiguous copies
    save_for_hf(
        model,
        tokenizer=tokenizer,
        model_config=model_config,
        save_directory=str(tmpdir),
        safe_serialization="both",
        skip_weights=True,
    )
    # We load configs from the vendored hlip/model_configs JSONs in README/examples,
    # so the generated open_clip_config.json is not needed in this repo layout.
    open_clip_cfg_path = tmpdir / "open_clip_config.json"
    if open_clip_cfg_path.exists():
        open_clip_cfg_path.unlink()

    # 6b) Save contiguous weights
    print("      Saving contiguous weights (.safetensors and .bin)")
    sd_contig = {k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}
    safetensors_path = tmpdir / "model.safetensors"
    bin_path = tmpdir / "pytorch_model.bin"
    safetensors.torch.save_file(sd_contig, safetensors_path)
    torch.save(sd_contig, bin_path)

    # 6c) Vendor your custom code (hlip/)
    print("      Vendoring custom package: hlip/")
    vendor_namespace_package("hlip", tmpdir)

    # 6d) Add README and metadata config.json for Hub UX/download tracking
    print("      Writing README.md and config.json")
    (tmpdir / "README.md").write_text(build_readme(repo_id, model_name, docs_src_dir.name))
    (tmpdir / "config.json").write_text(json.dumps({"date": publish_date}, indent=2) + "\n")

    # 6e) Add example docs payload
    docs_dst = tmpdir / "docs" / docs_src_dir.name
    print(f"      Copying docs sample to {docs_dst}")
    docs_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(docs_src_dir, docs_dst, dirs_exist_ok=True)

    # 6f) Optional: requirements for consumers
    (tmpdir / "requirements.txt").write_text(
        "open-clip-torch\n"
        "torch\n"
        "timm"
    )

    print(f"[7/7] Uploading folder to Hub: {repo_id}")
    upload_folder(
        repo_id=repo_id,
        folder_path=str(tmpdir),
        commit_message=args.commit_message,
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()

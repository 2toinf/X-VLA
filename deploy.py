# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

import argparse
import os
import os.path as osp
import json
import torch
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor
import sys


def get_default_device() -> torch.device:
    """Pick CUDA when present, then Intel XPU, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Launch XVLA inference FastAPI server")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained XVLA model directory")
    parser.add_argument('--processor_path', type=str, default=None)
    parser.add_argument('--LoRA_path', type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./logs",
                        help="Directory to save runtime info (info.json)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to load model on (auto / cuda / xpu / cpu); auto picks cuda, then xpu, then cpu")
    parser.add_argument("--port", default=8010, type=int,
                        help="Port number for FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", type=str,
                        help="Host address for FastAPI server")
    parser.add_argument("--disable_slurm", action="store_true", default=False)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("🚀 Starting XVLA Inference Server...")
    print(f"🔹 Model Path  : {args.model_path}")
    print(f"🔹 Output Dir  : {args.output_dir}")
    print(f"🔹 Device Arg  : {args.device}")
    print(f"🔹 Port        : {args.port}")

    # --------------------------------------------------------------------------
    # Select device automatically
    # --------------------------------------------------------------------------
    if args.device == "auto":
        device = get_default_device()
    else:
        device = torch.device(args.device)
    print(f"🧠 Using device: {device}")

    # --------------------------------------------------------------------------
    # Load processor (if available)
    # --------------------------------------------------------------------------
    processor = None
    try:
        print("\n🧩 Loading XVLAProcessor...")
        processor_path = args.processor_path if args.processor_path else args.model_path
        processor =  XVLAProcessor.from_pretrained(processor_path)
        print("✅ XVLAProcessor loaded successfully.")
    except Exception as e:
        print(f"⚠️ No processor found or failed to load: {e}")

    # --------------------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------------------
    print("\n📦 Loading XVLA model from pretrained checkpoint...")
    try:
        model = XVLA.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device).to(torch.float32)
        
        if args.LoRA_path is not None:
            print(f"🔸 Applying LoRA weights from {args.LoRA_path} ...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                args.LoRA_path,
                torch_dtype=torch.float32,
            ).to(device)
            
            print("✅ LoRA weights applied successfully.")
            
            
        print("✅ Model successfully loaded and moved to device.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # --------------------------------------------------------------------------
    # SLURM environment detection
    # --------------------------------------------------------------------------
    node_list = os.environ.get("SLURM_NODELIST")
    job_id = os.environ.get("SLURM_JOB_ID", "none")

    if node_list and not args.disable_slurm:
        print("\n🖥️  SLURM Environment Detected:")
        print(f"   Node list : {node_list}")
        print(f"   Job ID    : {job_id}")

        # Extract host
        try:
            host = ".".join(node_list.split("-")[1:]) if "-" in node_list else node_list
        except Exception:
            host = args.host
    else:
        print("\n⚠️  No SLURM environment detected, defaulting to 0.0.0.0")
        host = args.host

    # --------------------------------------------------------------------------
    # Write info.json for bookkeeping (safe version)
    # --------------------------------------------------------------------------
    info_path = osp.join(args.output_dir, "info.json")
    infos = {
        "host": host,
        "port": args.port,
        "job_id": job_id,
        "node_list": node_list or "none",
    }

    # --- Check existence before writing ---
    if osp.exists(info_path):
        print(f"❌ Error: {info_path} already exists. "
            f"This usually means another server is still running or the previous job did not clean up properly.")
        print("👉 Please remove it manually or use a different --output_dir.")
        sys.exit(1)

    # --- Write safely ---
    try:
        with open(info_path, "w") as f:
            json.dump(infos, f, indent=4)
        print(f"📝 Server info written to {info_path}")
    except Exception as e:
        print(f"⚠️ Failed to write {info_path}: {e}")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Launch FastAPI server
    # --------------------------------------------------------------------------
    print(f"\n🌐 Launching FastAPI service at http://{host}:{args.port} ...")
    try:
        if hasattr(model, "run"):
            model.run(processor=processor, host=host, port=args.port)
        else:
            print("❌ The loaded model does not implement `.run()` (FastAPI entrypoint).")
    except KeyboardInterrupt:
        print("\n🛑 Server stopped manually.")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")


if __name__ == "__main__":
    main()

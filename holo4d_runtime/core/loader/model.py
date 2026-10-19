from ..vram.initialization import skip_model_initialization
from ..vram.disk_map import DiskMap
from ..vram.layers import enable_vram_management
from .file import load_state_dict
import os
import torch


def _env_flag_enabled(name):
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _is_cpu_device(device):
    return str(device).startswith("cpu")


def _iter_state_dict_keys(state_dict):
    if hasattr(state_dict, "keys"):
        return state_dict.keys()
    return state_dict


def load_model(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, use_disk_map=False, module_map=None, vram_config=None, vram_limit=None):
    config = {} if config is None else config
    load_on_device = _env_flag_enabled("DIFFSYNTH_LOAD_ON_DEVICE")
    staging_device = device if load_on_device and not _is_cpu_device(device) else "cpu"
    # Why do we use `skip_model_initialization`?
    # It skips the random initialization of model parameters,
    # thereby speeding up model loading and avoiding excessive memory usage.
    with skip_model_initialization():
        model = model_class(**config)
    # What is `module_map`?
    # This is a module mapping table for VRAM management.
    if module_map is not None:
        devices = [vram_config["offload_device"], vram_config["onload_device"], vram_config["preparing_device"], vram_config["computation_device"]]
        device = [d for d in devices if d != "disk"][0]
        dtypes = [vram_config["offload_dtype"], vram_config["onload_dtype"], vram_config["preparing_dtype"], vram_config["computation_dtype"]]
        dtype = [d for d in dtypes if d != "disk"][0]
        if vram_config["offload_device"] != "disk":
            state_dict = DiskMap(path, staging_device, torch_dtype=dtype)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)

            # Set dtype on meta tensors first so `to_empty` allocates directly in target dtype.
            if dtype is not None:
                model = model.to(dtype=dtype)
            # Move model from meta device to staging device.
            model = model.to_empty(device=staging_device)

            # Load compatible parameters and buffers directly (manual copy to handle missing keys)
            model_params_buffers = dict(list(model.named_parameters()) + list(model.named_buffers()))
            missing_keys = []
            unexpected_keys = []

        for key in _iter_state_dict_keys(state_dict):
            if key not in model_params_buffers:
                unexpected_keys.append(key)

            # Load parameters to staging device first
            for key, param in model.named_parameters():
                if key in state_dict:
                    if param.shape == state_dict[key].shape:
                        param.data.copy_(state_dict[key].to(device=staging_device, dtype=dtype))
                    else:
                        print(f"Shape mismatch for {key}: model {param.shape} vs checkpoint {state_dict[key].shape}")
                else:
                    missing_keys.append(key)
                    # Initialize missing parameters (e.g., domain_embedding)
                    torch.nn.init.normal_(param.data, std=0.02)

            # Load buffers to staging device first
            for key, buffer in model.named_buffers():
                if key in state_dict:
                    if buffer.shape == state_dict[key].shape:
                        buffer.copy_(state_dict[key].to(device=staging_device, dtype=dtype))
                    else:
                        print(f"Shape mismatch for {key}: model {buffer.shape} vs checkpoint {state_dict[key].shape}")

            if missing_keys:
                print(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in checkpoint: {unexpected_keys}")

            # Now move to target dtype first, then device
            if dtype is not None:
                model = model.to(dtype=dtype)
            if str(device) != str(staging_device):
                model = model.to(device=device)

            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit)
        else:
            disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit)
    else:
        # Why do we use `DiskMap`?
        # Sometimes a model file contains multiple models,
        # and DiskMap can load only the parameters of a single model,
        # avoiding the need to load all parameters in the file.
        if use_disk_map:
            state_dict = DiskMap(path, staging_device, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict(path, torch_dtype, staging_device)
        # Why do we use `state_dict_converter`?
        # Some models are saved in complex formats,
        # and we need to convert the state dict into the appropriate format.
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)

        # Set dtype on meta tensors first so `to_empty` allocates directly in target dtype.
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        # Move model from meta device to staging device.
        model = model.to_empty(device=staging_device)

        # Load compatible parameters and buffers directly (manual copy to handle missing keys)
        model_params_buffers = dict(list(model.named_parameters()) + list(model.named_buffers()))
        missing_keys = []
        unexpected_keys = []

        for key in _iter_state_dict_keys(state_dict):
            if key not in model_params_buffers:
                unexpected_keys.append(key)

        # Load parameters to staging device first
        for key, param in model.named_parameters():
            if key in state_dict:
                if param.shape == state_dict[key].shape:
                    param.data.copy_(state_dict[key].to(device=staging_device, dtype=torch_dtype))
                else:
                    print(f"Shape mismatch for {key}: model {param.shape} vs checkpoint {state_dict[key].shape}")
            else:
                missing_keys.append(key)
                # Initialize missing parameters (e.g., domain_embedding)
                torch.nn.init.normal_(param.data, std=0.02)

        # Load buffers to staging device first
        for key, buffer in model.named_buffers():
            if key in state_dict:
                if buffer.shape == state_dict[key].shape:
                    buffer.copy_(state_dict[key].to(device=staging_device, dtype=torch_dtype))
                else:
                    print(f"Shape mismatch for {key}: model {buffer.shape} vs checkpoint {state_dict[key].shape}")

        if missing_keys:
            print(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in checkpoint: {unexpected_keys}")

        # Now move to target device/dtype (this will be handled by the caller or later)
        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)
        if str(device) != str(staging_device):
            model = model.to(device=device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def load_model_with_disk_offload(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, module_map=None):
    if isinstance(path, str):
        path = [path]
    config = {} if config is None else config
    with skip_model_initialization():
        model = model_class(**config)
    if hasattr(model, "eval"):
        model = model.eval()
    disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": device,
        "computation_dtype": torch_dtype,
        "computation_device": device,
    }
    enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=80)
    return model

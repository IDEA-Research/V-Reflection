import re
import os
import torch

from qwen_vl_utils import process_vision_info

from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,

    LVR_START_TOKEN,
    LVR_END_TOKEN,
    LVR_TOKEN,
    LVR_LATENT_END_TOKEN,
    LVR_PLACEHOLDER,

)


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def replace_lvr_tokens(input_string,lvr_token_idxs_list,latent_end_token,fixed_num_of_lvr_tokens,use_fixed_num_lvr_tokens=False):
    '''
    video not implemented

    Args:
        use_fixed_num_lvr_tokens: If True, uses hidden state passing mechanism (no actual <|lvr|> tokens)
                                 Only <|lvr_start|> and <|lvr_end|> markers are generated
                                 During inference: hidden state passing controlled by lvr_steps parameter
    '''
    pattern = r'\n?' + re.escape(LVR_PLACEHOLDER) + r'\n?'
    if re.search(pattern, input_string):
        input_segments = input_string.split(LVR_PLACEHOLDER)[1:]
        output_segments = []
        if use_fixed_num_lvr_tokens:
            # Fixed N LVR mode with hidden state passing:
            # Only generate <|lvr_start|> and <|lvr_end|> markers, NO <|lvr|> tokens in between
            # During training: use hidden state passing + MLP loss
            # During inference: hidden state passing controlled by lvr_steps=[16]
            for seg in input_segments:
                replacement = LVR_START_TOKEN + LVR_END_TOKEN  # No <|lvr|> tokens in between!
                output_segments.append(replacement+seg)
        elif fixed_num_of_lvr_tokens is not None:
            # Fixed N LVR tokens per segment (e.g. 8 for Box-Guided Compression); optional latent_end
            for seg in input_segments:
                if latent_end_token:
                    replacement = LVR_START_TOKEN + LVR_TOKEN*fixed_num_of_lvr_tokens + LVR_LATENT_END_TOKEN + LVR_END_TOKEN
                else:
                    replacement = LVR_START_TOKEN + LVR_TOKEN*fixed_num_of_lvr_tokens + LVR_END_TOKEN
                output_segments.append(replacement+seg)
        else:
            for seg,idxs in zip(input_segments,lvr_token_idxs_list):
                if latent_end_token is not None:    #latent end token mode will append a stopping token as the last
                    replacement = LVR_START_TOKEN + LVR_TOKEN*len(idxs) + LVR_LATENT_END_TOKEN + LVR_END_TOKEN
                else:
                    replacement = LVR_START_TOKEN + LVR_TOKEN*len(idxs) + LVR_END_TOKEN
                output_segments.append(replacement+seg)
        return "".join(output_segments)
    else:
        return input_string



def llava_to_openai_lvr(conversations, is_video=False, lvr_token_idxs_list=None, latent_end_token=False, fixed_num_of_lvr_tokens=None, use_fixed_num_lvr_tokens=False):

    # assert lvr_token_idxs_list is not None

    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_content = replace_lvr_tokens(transformed_content,lvr_token_idxs_list,latent_end_token,fixed_num_of_lvr_tokens,use_fixed_num_lvr_tokens)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel, width, height):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    
    # Check if image path exists (None means file not found)
    if image_path is None or not os.path.exists(image_path):
        return None

    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, width, height, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future
    content = {
        "type": "video", 
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "fps": fps
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {"role": "user", 
         "content": [content]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs


def map_image_path(image_path, image_folder, dataset_name=None):
    """
    Map image path from JSON format to actual file system path.
    
    JSON format: viscot/{dataset_name}/{path_to_file}
    Actual path: {image_folder}/{mapped_dataset_path}/{path_to_file}
    
    Args:
        image_path: Image path from JSON (e.g., "viscot/cub/001.Black_footed_Albatross/image.jpg")
        image_folder: Base image folder (e.g., "data/images" or absolute path to your image directory)
        dataset_name: Optional dataset name from JSON item, used as fallback if not in path
    
    Returns:
        Mapped absolute path to the image file
    """
    # If already an absolute path and exists, return as is
    if os.path.isabs(image_path) and os.path.exists(image_path):
        return image_path
    
    # If it's a URL, return as is
    if image_path.startswith("http"):
        return image_path
    
    # Dataset path mappings based on Visual_cot structure
    dataset_path_mappings = {
        'cub': 'cub/CUB_200_2011/images',
        'docvqa': 'docvqa',
        'flickr30k': 'flickr30k/flickr30k-images',
        'gqa': 'gqa/images',
        'infographicsvqa': 'infographicsvqa',
        'openimages': 'openimages',
        'textcap': 'textvqa/train_images',
        'textvqa': 'textvqa/train_images',
        'v7w': 'visual7w/images',
        'vsr': 'vsr/images',
        'sroie': 'sroie',
        'dude': 'dude',
        'coco': 'coco',
    }
    
    # Remove viscot/ prefix if present
    if image_path.startswith('viscot/'):
        image_path = image_path[7:]  # Remove 'viscot/'
    
    # Extract dataset name from path if not provided
    if dataset_name is None:
        path_parts = image_path.split('/', 1)
        if len(path_parts) > 1:
            dataset_name = path_parts[0]
            remaining_path = path_parts[1]
        else:
            # If no dataset name in path, try to use default mapping
            remaining_path = image_path
    else:
        # Remove dataset name from path if it's at the start
        if image_path.startswith(f'{dataset_name}/'):
            remaining_path = image_path[len(dataset_name)+1:]
        else:
            remaining_path = image_path
    
    # Get mapped dataset path
    mapped_dataset_path = dataset_path_mappings.get(dataset_name, dataset_name)
    
    # Special handling for cub dataset - preserve subdirectory structure
    if dataset_name == 'cub':
        # For cub, the path already includes subdirectory (e.g., "001.Black_footed_Albatross/image.jpg")
        full_path = os.path.join(image_folder, mapped_dataset_path, remaining_path)
    elif dataset_name == 'openimages':
        # For openimages, files are in subdirectories like train_0, train_1, etc.
        filename = os.path.basename(remaining_path)
        base_path = os.path.join(image_folder, mapped_dataset_path)
        
        # First try direct path
        full_path = os.path.join(base_path, filename)
        if os.path.exists(full_path):
            return full_path
        
        # Search in train_0 to train_5 subdirectories
        if os.path.exists(base_path):
            for subdir_idx in range(6):  # train_0 to train_5
                subdir = f'train_{subdir_idx}'
                subdir_path = os.path.join(base_path, subdir)
                if os.path.isdir(subdir_path):
                    candidate_path = os.path.join(subdir_path, filename)
                    if os.path.exists(candidate_path):
                        return candidate_path
        
        # Return None if not found (caller should handle this)
        return None
    elif dataset_name in ['dude', 'sroie']:
        # For dude and sroie, check multiple possible locations
        filename = os.path.basename(remaining_path)
        
        # Try viscot subdirectory first (common location)
        viscot_path = os.path.join(image_folder, 'viscot', dataset_name, filename)
        if os.path.exists(viscot_path):
            return viscot_path
        
        # Try direct dataset path
        direct_path = os.path.join(image_folder, mapped_dataset_path, filename)
        if os.path.exists(direct_path):
            return direct_path
        
        # For dude, also check DUDE_train-val-test_binaries/images/train/
        if dataset_name == 'dude':
            dude_train_path = os.path.join(image_folder, 'dude', 'DUDE_train-val-test_binaries', 'images', 'train', filename)
            if os.path.exists(dude_train_path):
                return dude_train_path
        
        # Return the viscot path as default (most likely location)
        return viscot_path
    else:
        # For other datasets, use only the filename
        filename = os.path.basename(remaining_path)
        full_path = os.path.join(image_folder, mapped_dataset_path, filename)
    
    return full_path
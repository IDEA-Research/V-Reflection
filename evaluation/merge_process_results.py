#!/usr/bin/env python3
"""
Merge multi-process evaluation results.
Finds all _process*.json files and merges them into a single JSON file.
"""
import os
import json
import sys
import glob
from pathlib import Path

def merge_process_results(result_dir, dataset_name, checkpoint_num, step_num):
    """
    Merge all _process*.json files for a given checkpoint and step.
    
    Args:
        result_dir: Directory containing the result files
        dataset_name: Name of the dataset (blink, vstar_bench, MMVP)
        checkpoint_num: Checkpoint number (e.g., "300")
        step_num: Step number (e.g., "4")
    
    Returns:
        True if merge was successful, False otherwise
    """
    # Construct the base filename pattern
    base_filename = f"ck-{checkpoint_num}-step{step_num}"
    final_file = os.path.join(result_dir, f"{base_filename}.json")
    
    # Find all process files
    process_pattern = os.path.join(result_dir, f"{base_filename}_process*.json")
    process_files = sorted(glob.glob(process_pattern))
    
    if not process_files:
        # No process files to merge
        return False
    
    print(f"[Merge] Found {len(process_files)} process files for {base_filename}")
    
    # Check if final file already exists and is complete
    if os.path.exists(final_file):
        try:
            with open(final_file, 'r') as f:
                existing_data = json.load(f)
            # Check if it's a complete result (has more than just summary)
            if isinstance(existing_data, list) and len(existing_data) > 0:
                # Check if it has actual results (not just summary)
                has_results = any('id' in item or 'prediction' in item for item in existing_data)
                if has_results:
                    print(f"[Merge] Final file {final_file} already exists and appears complete, skipping merge")
                    # Still clean up process files if they exist
                    for pf in process_files:
                        try:
                            os.remove(pf)
                            print(f"[Merge] Removed process file: {pf}")
                        except Exception as e:
                            print(f"[Merge] Warning: Failed to remove {pf}: {e}")
                    return True
        except Exception as e:
            print(f"[Merge] Warning: Could not read existing final file: {e}")
    
    # Merge all process files
    merged_result = []
    for process_file in process_files:
        try:
            with open(process_file, 'r') as f:
                process_data = json.load(f)
                if isinstance(process_data, list):
                    merged_result.extend(process_data)
                else:
                    print(f"[Merge] Warning: {process_file} does not contain a list, skipping")
        except Exception as e:
            print(f"[Merge] Error reading {process_file}: {e}")
            continue
    
    if not merged_result:
        print(f"[Merge] Warning: No data to merge for {base_filename}")
        return False
    
    # Sort by id to maintain order
    try:
        merged_result.sort(key=lambda x: int(x.get('id', 0)) if isinstance(x.get('id'), (str, int)) else 0)
    except Exception as e:
        print(f"[Merge] Warning: Could not sort results: {e}")
    
    # Import accuracy calculation function if available
    try:
        # Add project root to path
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import from evaluation.py in the same directory
        import importlib.util
        eval_module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation.py')
        spec = importlib.util.spec_from_file_location("evaluation", eval_module_path)
        eval_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_module)
        
        merged_result = eval_module.calculate_and_add_accuracy_summary(merged_result, final_file)
    except Exception as e:
        print(f"[Merge] Warning: Could not calculate accuracy summary: {e}")
    
    # Save merged result
    try:
        with open(final_file, 'w') as f:
            json.dump(merged_result, f, indent=2)
        print(f"[Merge] Successfully merged {len(merged_result)} results into {final_file}")
        
        # Clean up process files
        for process_file in process_files:
            try:
                os.remove(process_file)
                print(f"[Merge] Removed process file: {process_file}")
            except Exception as e:
                print(f"[Merge] Warning: Failed to remove {process_file}: {e}")
        
        return True
    except Exception as e:
        print(f"[Merge] Error saving merged result: {e}")
        return False

def main():
    if len(sys.argv) < 5:
        print("Usage: merge_process_results.py <result_dir> <dataset_name> <checkpoint_num> <step_num>")
        print("Example: merge_process_results.py /path/to/results MMVP 300 4")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    dataset_name = sys.argv[2]
    checkpoint_num = sys.argv[3]
    step_num = sys.argv[4]
    
    if not os.path.isdir(result_dir):
        print(f"Error: Result directory does not exist: {result_dir}")
        sys.exit(1)
    
    success = merge_process_results(result_dir, dataset_name, checkpoint_num, step_num)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()


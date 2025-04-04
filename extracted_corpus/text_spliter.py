import os
import argparse

def split_file(input_file, output_dir="split_files", chunk_size_mb=2, preserve_lines=True):
    """
    Split a large text file into smaller chunks.
    
    Args:
        input_file (str): Path to the input file to be split.
        output_dir (str): Directory where split files will be saved.
        chunk_size_mb (float): Size of each chunk in megabytes.
        preserve_lines (bool): If True, will not split in the middle of a line.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert MB to bytes
    chunk_size = int(chunk_size_mb * 1024 * 1024)
    
    # Get the base filename without extension
    base_name = os.path.basename(input_file)
    name_parts = os.path.splitext(base_name)
    base_name = name_parts[0]
    extension = name_parts[1] if len(name_parts) > 1 else ".txt"
    
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        file_number = 1
        chunk = []
        current_size = 0
        
        for line in f:
            line_size = len(line.encode('utf-8'))
            
            # If adding this line would exceed the chunk size and we have content,
            # write the chunk to a file
            if preserve_lines and current_size + line_size > chunk_size and chunk:
                # Write current chunk to file
                output_file = os.path.join(output_dir, f"{base_name}_part{file_number}{extension}")
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(''.join(chunk))
                
                print(f"Created {output_file} ({current_size / (1024*1024):.2f} MB)")
                
                # Reset for next chunk
                chunk = []
                current_size = 0
                file_number += 1
            
            # Add line to current chunk
            chunk.append(line)
            current_size += line_size
            
            # If not preserving lines, we can split in the middle of the buffer
            if not preserve_lines and current_size >= chunk_size:
                output_file = os.path.join(output_dir, f"{base_name}_part{file_number}{extension}")
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(''.join(chunk))
                
                print(f"Created {output_file} ({current_size / (1024*1024):.2f} MB)")
                
                # Reset for next chunk
                chunk = []
                current_size = 0
                file_number += 1
        
        # Write the last chunk if there's anything left
        if chunk:
            output_file = os.path.join(output_dir, f"{base_name}_part{file_number}{extension}")
            with open(output_file, 'w', encoding='utf-8') as out_f:
                out_f.write(''.join(chunk))
            
            print(f"Created {output_file} ({current_size / (1024*1024):.2f} MB)")
    
    print(f"Split complete! Created {file_number} files in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a large text file into smaller chunks")
    parser.add_argument("input_file", help="Path to the input file to be split")
    parser.add_argument("--output-dir", default="split_files", help="Directory to save split files (default: split_files)")
    parser.add_argument("--chunk-size", type=float, default=2.0, help="Size of each chunk in megabytes (default: 2.0)")
    parser.add_argument("--split-lines", action="store_false", dest="preserve_lines", 
                        help="Allow splitting in the middle of lines (default: preserve lines)")
    
    args = parser.parse_args()
    
    split_file(args.input_file, args.output_dir, args.chunk_size, args.preserve_lines)
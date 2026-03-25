import numpy as np
import h5py
import zipfile

def npz_headers(npz):
    """Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype) for each array in the archive.
    """
    with zipfile.ZipFile(npz) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue

            with archive.open(name) as npy:
                version = np.lib.format.read_magic(npy)
                shape, _, dtype = np.lib.format._read_array_header(npy, version)
                yield name[:-4], shape, dtype

# List of .npz file names
npz_files = ['50m_part1.npz', '50m_part2.npz', '50m_part3.npz', '50m_part4.npz']

print("Calculating total shapes and data types...")

# Initialize variables to calculate total shape and determine dtype for images and labels
total_images_shape = None
total_labels_shape = None
image_dtype = None
label_dtype = None
total_images_samples = 0
total_labels_samples = 0

# Use npz_headers to retrieve shapes and dtypes efficiently
for npz_file in npz_files:
    for name, shape, dtype in npz_headers(npz_file):
        if name == 'image':
            total_images_samples += shape[0]
            if total_images_shape is None:
                total_images_shape = list(shape)
                image_dtype = dtype
        elif name == 'label':
            total_labels_samples += shape[0]
            if total_labels_shape is None:
                total_labels_shape = list(shape)
                label_dtype = dtype
    print(0)
# Set the total number of samples for the combined shape
total_images_shape[0] = total_images_samples
total_labels_shape[0] = total_labels_samples

print("Total shapes calculated:")
print(f"  Images shape: {total_images_shape}, dtype: {image_dtype}")
print(f"  Labels shape: {total_labels_shape}, dtype: {label_dtype}")

print("Creating HDF5 file and datasets...")
# Create the HDF5 file and datasets with combined shapes and original dtypes
with h5py.File('50m_combined.h5', 'w') as f_out:
    dset_image = f_out.create_dataset('image', shape=total_images_shape, dtype=image_dtype)
    dset_label = f_out.create_dataset('label', shape=total_labels_shape, dtype=label_dtype)

    current_index = 0  # Tracks position for each part's data within the combined dataset

    for i, npz_file in enumerate(npz_files):
        print(f"Loading data from {npz_file}...")
        with np.load(npz_file) as part:
            # Determine the number of samples in this part
            num_samples = part['image'].shape[0]
            
            print(f"  Writing images and labels for part {i+1} at index {current_index}...")
            # Write this part's images and labels to the combined datasets
            dset_image[current_index:current_index + num_samples] = part['image']
            dset_label[current_index:current_index + num_samples] = part['label']
            
            # Update the index
            current_index += num_samples
            print(f"  Finished writing part {i+1}. Current index is now {current_index}.")

print("All parts have been combined and written to '50m_combined.h5'.")

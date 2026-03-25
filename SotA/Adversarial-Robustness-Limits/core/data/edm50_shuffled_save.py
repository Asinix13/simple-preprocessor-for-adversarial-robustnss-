import os
import h5py
import numpy as np
import time


def get_shuffled_EDM_50_indices():
    indices_path = "../../mnt/MLdata/virtual/EDM_50_indices.npy"
    if not os.path.exists(indices_path):
        EDM_50_indices = np.arange(int(50e6))
        np.random.shuffle(EDM_50_indices)
        np.save(indices_path, EDM_50_indices)
        return EDM_50_indices
    return np.load(indices_path)

def main(name, chunk_size = 100000, tmp_saves = 500):  
    arr = np.arange(int(chunk_size))
    tmp_name_beg = '../../mnt/MLdata/virtual/edm50_temp'
    tmp_size = chunk_size//tmp_saves


    print('Opening HDF5 file...')
    with h5py.File("../../mnt/MLdata/CIFAR-10-EDM/50m.h5", "r", libver='latest', swmr=True) as aux:
        num_samples = aux['image'].shape[0]
        tmpi_shape = list(aux['image'].shape)
        tmpl_shape = list(aux['label'].shape)

        tmpi_shape[0] = num_samples // tmp_saves 
        tmpl_shape[0] = num_samples // tmp_saves

        for i in range(0, num_samples, chunk_size):
            image_chunk = aux['image'][i:i+chunk_size]
            label_chunk = aux['label'][i:i+chunk_size]
            np.random.shuffle(arr)
            label_chunk = label_chunk[arr]
            image_chunk = image_chunk[arr]

            for j in range(tmp_saves): 
                tmp_name = tmp_name_beg + str(j) + '.h5'
                with h5py.File(tmp_name, "w", libver='latest') as f_out:
                    image_dset = f_out.create_dataset('image', tmpi_shape, dtype=np.uint8)
                    label_dset = f_out.create_dataset('label', tmpl_shape, dtype=np.uint8)
                    image_dset[i*tmp_size:(i + 1)*tmp_size] = image_chunk[j*tmp_size:(j + 1)*tmp_size]
                    label_dset[i*tmp_size:(i + 1)*tmp_size] = label_chunk[j*tmp_size:(j + 1)*tmp_size]
              
            print(i) 

        arr = np.arange(tmpi_shape[0])
        if not os.path.exists(name):
            with h5py.File(name, "w", libver='latest') as f_out:
                image_dset = f_out.create_dataset('image', shape=aux['image'].shape, dtype=np.uint8)
                label_dset = f_out.create_dataset('label', shape=aux['label'].shape, dtype=np.uint8)

                for i in range(tmp_saves):
                    tmp_name = tmp_name_beg + str(i) + '.h5'

                    with h5py.File(tmp_name, "r", libver='latest', swmr=True) as aux:
                        label_chunk = aux['label'][:]
                        image_chunk = aux['image'][:]
                        np.random.shuffle(arr)
                    # Store in the original shuffled order
                    image_dset[i*tmpi_shape[0]:(i + 1)*tmpi_shape[0]] = image_chunk[arr]
                    label_dset[i*tmpi_shape[0]:(i + 1)*tmpi_shape[0]] = label_chunk[arr]
                    print(i)
        else:
            assert False, f'The file {name} exists! Skipping creation.'

    print('Shuffling complete.')

if __name__=='__main__':
    name = '../../mnt/MLdata/virtual/edm50_shuffled.h5'  
    if not os.path.exists(name):
        main(name)
    else:
        print(f'The file {name} exists! Skipping creation.')
    
import hashlib
import pdb
import time

def md5(fname, blocksize=65536):
    """
    Performs md5 on file
    Parameters
    ----------
    fname : str
        path to the file
    blocksize : int
        4096 or 65536
    Returns
    -------

    """

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(blocksize), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def check_copied_file_identical(original_fpath, copied_fpath, verbose=True):
    """
    Checks if two files are identical (and not unintentionally modified through eg. a copy and paste process)
    Parameters
    ----------
    original_fpath : str
        path to the original file
    copied_fpath : str
        path to the new copied file

    Returns
    -------
    file_is_identical : bool
        0 : file is different
        1 : file is identical
    """

    if verbose:
        print('Calculating md5 for %s' % original_fpath)
        start_time = time.time()
    original_md5 = md5(original_fpath)

    if verbose:
        end_time = time.time()
        print('Elapsed time %.3f' % (end_time - start_time))

    if verbose:
        print('Calculating md5 for %s' % copied_fpath)
        start_time = time.time()

    copied_md5 = md5(copied_fpath)

    if verbose:
        end_time = time.time()
        print('Elapsed time %.3f' % (end_time - start_time))

    file_is_identical = (original_md5 == copied_md5)

    if verbose:
        if file_is_identical:
            print('Files are identical')
        else:
            print('Files are different')

    return file_is_identical



def main():


    original_fpath = '/home/timothysit/Desktop/2022-01-18_6_AV002_frontCam.mj2'
    copied_fpath = '/run/user/1000/gvfs/smb-share:server=zinu.local,share=subjects/AV002/2022-01-18/6/2022-01-18_6_AV002_frontCam.mj2'

    file_is_identical = check_copied_file_identical(original_fpath, copied_fpath)


if __name__ == '__main__':
    main()

import tempfile
import os
import shutil


class SmartTemporaryDirectory(tempfile.TemporaryDirectory):
    def __init__(self, required_bytes, dir=None, verbose=False, *args, **kwargs):
        required_space = required_bytes / 1024 ** 3

        temp_dir_candidates = dir
        if temp_dir_candidates is None:
            import platform
            shm = ['/dev/shm'] if platform.system() == 'Linux' else []
            temp_dir_candidates = shm + [tempfile.gettempdir(), os.getcwd()]

        temp_dir = None
        for candidate in temp_dir_candidates:
            free_space = shutil.disk_usage(candidate)[2] / 1024 ** 3
            if os.path.exists(candidate) and os.access(candidate, os.W_OK) and free_space >= required_space * 1.2:
                temp_dir = candidate
                if verbose:
                    print(f'Using {temp_dir} as temporary directory.')
                break
        if temp_dir is None:
            raise OSError('Insufficient disk space.')

        super().__init__(dir=temp_dir, *args, **kwargs) 


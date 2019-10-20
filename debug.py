from fastai.torch_core import *
from fastai.data_block import *
from fastai.vision.image import *
from fastai.vision.data import *
import PIL

image_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('image/'))

def get_image_files(c:PathOrStr, check_ext:bool=True, recurse=False)->FilePathList:
    "Return list of files in `c` that are images. `check_ext` will filter to `image_extensions`."
    return get_files(c, extensions=(image_extensions if check_ext else None), recurse=recurse)

def verify_image(file:Path, idx:int, delete:bool, max_size:Union[int,Tuple[int,int]]=None, dest:Path=None, n_channels:int=3,
                 interp=PIL.Image.BILINEAR, ext:str=None, img_format:str=None, resume:bool=False, **kwargs):
    "Check if the image in `file` exists, maybe resize it and copy it in `dest`."
    try:
        # deal with partially broken images as indicated by PIL warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                with open(file, 'rb') as img_file: PIL.Image.open(img_file)
            except Warning as w:
                if "Possibly corrupt EXIF data" in str(w):
                    if delete: # green light to modify files
                        print(f"{file}: Removing corrupt EXIF data")
                        warnings.simplefilter("ignore")
                        # save EXIF-cleaned up image, which happens automatically
                        PIL.Image.open(file).save(file)
                    else: # keep user's files intact
                        print(f"{file}: Not removing corrupt EXIF data, pass `delete=True` to do that")
                else: warnings.warn(w)

        img = PIL.Image.open(file)
        imgarr = np.array(img)
        img_channels = 1 if len(imgarr.shape) == 2 else imgarr.shape[2]
        if (max_size is not None and (img.height > max_size or img.width > max_size)) or img_channels != n_channels:
            assert isinstance(dest, Path), "You should provide `dest` Path to save resized image"
            dest_fname = dest/file.name
            if ext is not None: dest_fname=dest_fname.with_suffix(ext)
            if resume and os.path.isfile(dest_fname): return
            if max_size is not None:
                new_sz = resize_to(img, max_size)
                img = img.resize(new_sz, resample=interp)
            if n_channels == 3: img = img.convert("RGB")
            # img.save(dest_fname, img_format, **kwargs)
    except Exception as e:
        print("[ERR]" , file)
        print(f'{e}')
        if delete: file.unlink()
    

def verify_images(path:PathOrStr, delete:bool=True, max_workers:int=10, max_size:Union[int]=None, recurse:bool=False,
                  dest:PathOrStr='.', n_channels:int=3, interp=PIL.Image.BILINEAR, ext:str=None, img_format:str=None,
                  resume:bool=None, **kwargs):
    "Check if the images in `path` aren't broken, maybe resize them and copy it in `dest`."
    path = Path(path)
    if resume is None and dest == '.': resume=False
    dest = path/Path(dest)
    #os.makedirs(dest, exist_ok=True)
    files = get_image_files(path, recurse=recurse)
    func = partial(verify_image, delete=delete, max_size=max_size, dest=dest, n_channels=n_channels, interp=interp,
                   ext=ext, img_format=img_format, resume=resume, **kwargs)
    parallel(func, files, max_workers=max_workers)

    
verify_images('/home/deva/DIV2K/DIV2K_train_HR_crop_128',delete=False)
verify_images('/home/deva/DIV2K/DIV2K_train_RR6a_crop_128',delete=False)

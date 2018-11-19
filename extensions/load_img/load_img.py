
load_img_type = 0
def set_load_img_type(lit):
    load_img_type = lit
    if load_img_type == 0:
        print("INFO: load_img_type = %d -> 'load_img' calls 'load_img_keras'" % load_img_type)
    if load_img_type == 1:
        print("INFO: load_img_type = %d -> 'load_img' calls 'load_img_pad'" % load_img_type)
    if load_img_type == 2:
        print("INFO: load_img_type = %d -> 'load_img' calls 'load_img_pad' (+mask)" % load_img_type)
    if load_img_type == 3:
        print("INFO: load_img_type = %d -> 'load_img' calls 'load_img_crop'" % load_img_type)
    if load_img_type == 4:
        print("INFO: load_img_type = %d -> 'load_img' calls 'load_img_multicrop'" % load_img_type)

def load_img(path, grayscale=False, target_size=None, color_mode='rgb', interpolation='nearest', min_dimension=224):    
    if load_img_type == 0:
        # load_img_keras (original method)
        return load_img_keras(path, grayscale=grayscale, color_mode=color_mode, target_size=target_size, interpolation=interpolation, min_dimension=min_dimension)
    if load_img_type == 1:
        # load_img_pad
        try:
            if (target_size[0] is None) or (target_size[1] is None):
                target_size=(min_dimension,min_dimension)
        except:
            target_size=(min_dimension,min_dimension)
        img, mask = load_img_pad(path, grayscale=grayscale, color_mode=color_mode, target_size=target_size, interpolation=interpolation, min_dimension=min_dimension)
        return img
    if load_img_type == 2:
        # load_img_pad (+mask)
        try:
            if (target_size[0] is None) or (target_size[1] is None):
                target_size=(min_dimension,min_dimension)
        except:
            target_size=(min_dimension,min_dimension)
        img, mask = load_img_pad(path, grayscale=grayscale, color_mode=color_mode, target_size=target_size, interpolation=interpolation, min_dimension=min_dimension)
        return img, mask
    if load_img_type == 3:
        # load_img_crop
        try:
            if (target_size[0] is None) or (target_size[1] is None):
                target_size=(min_dimension,min_dimension)
        except:
            target_size=(min_dimension,min_dimension)
        return load_img_crop(path, grayscale=grayscale, color_mode=color_mode, target_size=target_size, interpolation=interpolation, min_dimension=min_dimension)
    if load_img_type == 4:
        # load_img_multicrop
        try:
            if (target_size[0] is None) or (target_size[1] is None):
                target_size=(min_dimension,min_dimension)
        except:
            target_size=(min_dimension,min_dimension)
        return load_img_multicrop(path, grayscale=grayscale, color_mode=color_mode, target_size=target_size, interpolation=interpolation, min_dimension=min_dimension)
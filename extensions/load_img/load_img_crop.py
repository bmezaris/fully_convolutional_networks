
def load_img_crop(path, grayscale=False, color_mode='rgb', target_size=None, interpolation='nearest', min_dimension=224):
    img = pil_image.open(path)
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rbg", or "rgba"')
    desired_size = target_size[0]
    ratio = float(desired_size)/min(img.size)
    new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
    img = img.resize(new_size, _PIL_INTERPOLATION_METHODS[interpolation])
    img = img.crop(((new_size[0]-desired_size)/2, (new_size[1]-desired_size)/2,(new_size[0]+desired_size)/2,(new_size[1]+desired_size)/2))
    return img

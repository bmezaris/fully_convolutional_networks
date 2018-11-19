
def load_img_keras(path, grayscale=False, color_mode='rgb', target_size=None, interpolation='nearest', min_dimension=224):
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
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
            
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            if width_height_tuple[0] is None and width_height_tuple[1] is None:
                return img

            if width_height_tuple[0] is None:
                ratio = float(width_height_tuple[1]) / float(img.size[1])
                width_height_tuple = (round(img.size[0]*ratio), width_height_tuple[1])
                smaller_dim = min(width_height_tuple[0],width_height_tuple[1])
                if smaller_dim < min_dimension:
                    upscale_ratio = float(min_dimension) / float(smaller_dim)
                    width_height_tuple = (round(float(width_height_tuple[0])*upscale_ratio), round(float(width_height_tuple[1])*upscale_ratio))

            if width_height_tuple[1] is None:
                ratio = float(width_height_tuple[0]) / float(img.size[0])
                width_height_tuple = (width_height_tuple[0], round(img.size[1]*ratio))
                smaller_dim = min(width_height_tuple[0],width_height_tuple[1])
                if smaller_dim < min_dimension:
                    upscale_ratio = float(min_dimension) / float(smaller_dim)
                    width_height_tuple = (round(float(width_height_tuple[0])*upscale_ratio), round(float(width_height_tuple[1])*upscale_ratio))

            img = img.resize(width_height_tuple, resample)
    return img
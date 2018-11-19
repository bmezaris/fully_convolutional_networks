
def load_img_pad(path, grayscale=False, color_mode='rgb', target_size=None, interpolation='nearest', min_dimension=224):
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
    ratio = float(desired_size)/max(img.size)
    new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
    img = img.resize(new_size, _PIL_INTERPOLATION_METHODS[interpolation])
    # create a new image and paste the resized on it
    new_im = pil_image.new("RGB", (desired_size, desired_size))
    new_im.paste(img, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
    new_im_mask = pil_image.new("L", (desired_size, desired_size))
    im_mask = pil_image.new("L", new_size, color=255)
    new_im_mask.paste(im_mask, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
    return new_im, new_im_mask
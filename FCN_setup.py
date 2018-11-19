from __future__ import print_function

import sys
import os
import site
from shutil import copyfile

if len(sys.argv)<2:
	print("Usage:")
	print(" 'python", sys.argv[0], "install' to install FCN extensions to Keras")
	print("    or ")
	print(" 'python", sys.argv[0], "uninstall' to remove FCN extensions from Keras")
	sys.exit()
	
if sys.argv[1]=='install':
	mode = 1
elif sys.argv[1]=='uninstall':
	mode = 2
else:
	print("Usage:")
	print(" 'python", sys.argv[0], "install' to install FCN extensions to Keras")
	print("    or ")
	print(" 'python", sys.argv[0], "uninstall' to remove FCN extensions from Keras")
	sys.exit()

# site-packages path
site_path = ''
for p in site.getsitepackages():
	if "packages" in p:
		site_path = p
		break

# Keras paths
keras_path = os.path.join(site_path, 'keras')
keras_pre_path = os.path.join(site_path, 'keras', 'preprocessing')
keras_app_path = os.path.join(site_path, 'keras', 'applications')
keras_pre_pkg_path = os.path.join(site_path, 'keras_preprocessing')
keras_app_pkg_path = os.path.join(site_path, 'keras_applications')

# Check if keras' paths exist
if not os.path.isdir(keras_path):
	print('path', keras_path, 'not found.')
	print('Cannot continue...')
	sys.exit()

if not os.path.isdir(keras_pre_path):
	print('path', keras_pre_path, 'not found.')
	print('Cannot continue...')
	sys.exit()
	
if not os.path.isdir(keras_app_path):
	print('path', keras_app_path, 'not found.')
	print('Cannot continue...')
	sys.exit()
	
if not os.path.isdir(keras_pre_pkg_path):
	print('path', keras_pre_pkg_path, 'not found.')
	print('Cannot continue...')
	sys.exit()
	
if not os.path.isdir(keras_app_pkg_path):
	print('path', keras_app_pkg_path, 'not found.')
	print('Cannot continue...')
	sys.exit()
	
print('~~~ Found Keras in:')
print('   ', keras_path)
print('   ', keras_pre_path)
print('   ', keras_app_path)
print('   ', keras_pre_pkg_path)
print('   ', keras_app_pkg_path)
print(' ')

if mode==1:
	# check if Keras is installed
	print('~~~ Starting Keras')
	import keras
	print('Keras version:', keras.__version__)
	print(' ')
	print('Installing:')

	# write new content in keras/applications/__init__.py
	print('~~~ Checking keras/applications/__init__.py')
	if not os.path.isfile(os.path.join(keras_app_path, '__init__-backup')):
		print('    Creating backup of:', os.path.join(keras_app_path, '__init__.py'))
		copyfile(os.path.join(keras_app_path, '__init__.py'), os.path.join(keras_app_path, '__init__-backup'))
		
		print('    Writing new lines in', os.path.join(keras_app_path, '__init__.py'))
		with open(os.path.join(keras_app_path, '__init__.py'), 'a') as f:
			f.write('\n')
			f.write('from .vgg16_fcn import VGG16_fcn\n')
			f.write('from .vgg19_fcn import VGG19_fcn\n')
			f.write('from .inception_v3_fcn import InceptionV3_fcn\n')
			f.write('from .xception_fcn import Xception_fcn\n')
	else:
		print('    Skipped,', os.path.join(keras_app_path, '__init__.py'), 'is updated.)')
	print(' ')
		
	# write new content in keras_preprocessing/image.py
	print('~~~ Checking keras_preprocessing/image.py')
	if not os.path.isfile(os.path.join(keras_pre_pkg_path, 'image-backup')):
		print('    Creating backup of:', os.path.join(keras_pre_pkg_path, 'image.py'))
		copyfile(os.path.join(keras_pre_pkg_path, 'image.py'), os.path.join(keras_pre_pkg_path, 'image-backup'))
		
		print('    Writing new lines in', os.path.join(keras_pre_pkg_path, 'image.py'))
		with open(os.path.join(keras_pre_pkg_path, 'image.py'), 'r') as f:
			data = f.readlines()
			
		for i,line in enumerate(data):
			if 'load_img' in line:
				adjusted_line = line.replace('load_img', 'load_img_old')
				print('    (original line: "', line.strip(), '")')
				print('    (adjusted line: "', adjusted_line.strip(), '")')
				data[i] = adjusted_line
				break
				
		with open(os.path.join(keras_pre_pkg_path, 'image.py'), 'w') as f:
			for item in data:
				f.write("%s" % item)
		print(' ')

		print("~~~ Adding new content:")
		with open(os.path.join(keras_pre_pkg_path, 'image.py'), 'a') as f_dst:
			print("    adding method for 'load_img_keras'")
			with open(os.path.join('extensions', 'load_img', 'load_img_keras.py'), 'r') as f_src:
				lines = f_src.readlines()
				for line in lines:
					f_dst.write(line)
					
			print("    adding method for 'load_img_pad'")
			with open(os.path.join('extensions', 'load_img', 'load_img_pad.py'), 'r') as f_src:
				lines = f_src.readlines()
				for line in lines:
					f_dst.write(line)

			print("    adding method for 'load_img_crop'")
			with open(os.path.join('extensions', 'load_img', 'load_img_crop.py'), 'r') as f_src:
				lines = f_src.readlines()
				for line in lines:
					f_dst.write(line)
				
			print("    adding method for 'load_img_multicrop'")
			with open(os.path.join('extensions', 'load_img', 'load_img_multicrop.py'), 'r') as f_src:
				lines = f_src.readlines()
				for line in lines:
					f_dst.write(line)
				
			print("    adding method for top-level 'load_img'")
			with open(os.path.join('extensions', 'load_img', 'load_img.py'), 'r') as f_src:
				lines = f_src.readlines()
				for line in lines:
					f_dst.write(line)
	else:
		print('    Skipped,', os.path.join(keras_pre_path, 'image.py'), 'is updated.)')
	print(' ')
		

	# write new content in keras/preprocessing/image.py
	print('~~~ Checking keras/preprocessing/image.py')
	if not os.path.isfile(os.path.join(keras_pre_path, 'image-backup')):
		print('    Creating backup of:', os.path.join(keras_pre_path, 'image.py'))
		copyfile(os.path.join(keras_pre_path, 'image.py'), os.path.join(keras_pre_path, 'image-backup'))
		
		print('    Writing new lines in', os.path.join(keras_pre_path, 'image.py'))
		with open(os.path.join(keras_pre_path, 'image.py'), 'a') as f:	
			f.write('set_load_img_type = image.set_load_img_type\n')
	else:
		print('    Skipped,', os.path.join(keras_pre_path, 'image.py'), 'is updated.)')
	print(' ')



	# copy FCNs
	print('~~~ Copying FCNs to keras_applications:')
	if not os.path.isfile(os.path.join(keras_app_pkg_path, 'vgg16_fcn.py')):
		print('    copying', os.path.join(keras_app_pkg_path, 'vgg16_fcn.py'))
		copyfile(os.path.join('extensions', 'vgg16_fcn.py'), os.path.join(keras_app_pkg_path, 'vgg16_fcn.py'))
	else:
		print('    Skipped,', os.path.join(keras_app_pkg_path, 'vgg16_fcn.py') , 'exists')

	if not os.path.isfile(os.path.join(keras_app_pkg_path, 'vgg19_fcn.py')):	
		print('    copying', os.path.join(keras_app_pkg_path, 'vgg19_fcn.py'))
		copyfile(os.path.join('extensions', 'vgg19_fcn.py'), os.path.join(keras_app_pkg_path, 'vgg19_fcn.py'))
	else:
		print('    Skipped,', os.path.join(keras_app_pkg_path, 'vgg19_fcn.py') , 'exists')
		
	if not os.path.isfile(os.path.join(keras_app_pkg_path, 'inception_v3_fcn.py')):	
		print('    copying', os.path.join(keras_app_pkg_path, 'inception_v3_fcn.py'))
		copyfile(os.path.join('extensions', 'inception_v3_fcn.py'), os.path.join(keras_app_pkg_path, 'inception_v3_fcn.py'))
	else:
		print('    Skipped,', os.path.join(keras_app_pkg_path, 'inception_v3_fcn.py') , 'exists')

	if not os.path.isfile(os.path.join(keras_app_pkg_path, 'xception_fcn.py')):	
		print('    copying', os.path.join(keras_app_pkg_path, 'xception_fcn.py'))
		copyfile(os.path.join('extensions', 'xception_fcn.py'), os.path.join(keras_app_pkg_path, 'xception_fcn.py'))
	else:
		print('    Skipped,', os.path.join(keras_app_pkg_path, 'xception_fcn.py') , 'exists')
	print(' ')


	print('~~~ Copying FCNs to keras.applications:')
	if not os.path.isfile(os.path.join(keras_app_path, 'vgg16_fcn.py')):	
		print('    copying', os.path.join(keras_app_path, 'vgg16_fcn.py'))
		copyfile(os.path.join('extensions', 'keras.applications', 'vgg16_fcn.py'), os.path.join(keras_app_path, 'vgg16_fcn.py'))
	else:
		print('    Skipped,', os.path.join(keras_app_pkg_path, 'vgg16_fcn.py') , 'exists')

	if not os.path.isfile(os.path.join(keras_app_path, 'vgg19_fcn.py')):	
		print('    copying', os.path.join(keras_app_path, 'vgg19_fcn.py'))
		copyfile(os.path.join('extensions', 'keras.applications', 'vgg19_fcn.py'), os.path.join(keras_app_path, 'vgg19_fcn.py'))
	else:
		print('    Skipped,', os.path.join(keras_app_pkg_path, 'vgg19_fcn.py') , 'exists')

	if not os.path.isfile(os.path.join(keras_app_path, 'inception_v3_fcn.py')):	
		print('    copying', os.path.join(keras_app_path, 'inception_v3_fcn.py'))
		copyfile(os.path.join('extensions', 'keras.applications', 'inception_v3_fcn.py'), os.path.join(keras_app_path, 'inception_v3_fcn.py'))
	else:
		print('    Skipped,', os.path.join(keras_app_pkg_path, 'inception_v3_fcn.py') , 'exists')

	if not os.path.isfile(os.path.join(keras_app_path, 'xception_fcn.py')):	
		print('    copying', os.path.join(keras_app_path, 'xception_fcn.py'))
		copyfile(os.path.join('extensions', 'keras.applications', 'xception_fcn.py'), os.path.join(keras_app_path, 'xception_fcn.py'))
	else:
		print('    Skipped,', os.path.join(keras_app_pkg_path, 'xception_fcn.py') , 'exists')

	print('\nSuccessfully installed FCN extensions to Keras.')

		
if mode==2:
	print('Uninstalling:')
	
	# Restoring keras/applications/__init__.py
	print('~~~ Restoring keras/applications/__init__.py...')
	if os.path.isfile(os.path.join(keras_app_path, '__init__-backup')):
		os.remove(os.path.join(keras_app_path, '__init__.py'))
		copyfile(os.path.join(keras_app_path, '__init__-backup'), os.path.join(keras_app_path, '__init__.py'))
		os.remove(os.path.join(keras_app_path, '__init__-backup'))
		
	# Restoring keras_preprocessing/image.py
	print('~~~ Restoring keras_preprocessing/image.py...')
	if os.path.isfile(os.path.join(keras_pre_pkg_path, 'image-backup')):
		os.remove(os.path.join(keras_pre_pkg_path, 'image.py'))
		copyfile(os.path.join(keras_pre_pkg_path, 'image-backup'), os.path.join(keras_pre_pkg_path, 'image.py'))
		os.remove(os.path.join(keras_pre_pkg_path, 'image-backup'))
	
	# write new content in keras/preprocessing/image.py
	print('~~~ Restoring keras/preprocessing/image.py...')
	if os.path.isfile(os.path.join(keras_pre_path, 'image-backup')):
		os.remove(os.path.join(keras_pre_path, 'image.py'))
		copyfile(os.path.join(keras_pre_path, 'image-backup'), os.path.join(keras_pre_path, 'image.py'))
		os.remove(os.path.join(keras_pre_path, 'image-backup'))

	# Delete FCNs from keras_applications
	print('~~~ Removing FCNs from keras_applications...')
	if os.path.isfile(os.path.join(keras_app_pkg_path, 'vgg16_fcn.py')):
		print('    deleting', os.path.join(keras_app_pkg_path, 'vgg16_fcn.py'))
		os.remove(os.path.join(keras_app_pkg_path, 'vgg16_fcn.py'))

	if os.path.isfile(os.path.join(keras_app_pkg_path, 'vgg19_fcn.py')):	
		print('    deleting', os.path.join(keras_app_pkg_path, 'vgg19_fcn.py'))
		os.remove(os.path.join(keras_app_pkg_path, 'vgg19_fcn.py'))
		
	if os.path.isfile(os.path.join(keras_app_pkg_path, 'inception_v3_fcn.py')):	
		print('    deleting', os.path.join(keras_app_pkg_path, 'inception_v3_fcn.py'))
		os.remove(os.path.join(keras_app_pkg_path, 'inception_v3_fcn.py'))

	if os.path.isfile(os.path.join(keras_app_pkg_path, 'xception_fcn.py')):	
		print('    deleting', os.path.join(keras_app_pkg_path, 'xception_fcn.py'))
		os.remove(os.path.join(keras_app_pkg_path, 'xception_fcn.py'))

	# Delete FCNs from keras.applications
	print('~~~ Removing FCNs from keras.applications...')
	if os.path.isfile(os.path.join(keras_app_path, 'vgg16_fcn.py')):	
		print('    deleting', os.path.join(keras_app_path, 'vgg16_fcn.py'))
		os.remove(os.path.join(keras_app_path, 'vgg16_fcn.py'))

	if os.path.isfile(os.path.join(keras_app_path, 'vgg19_fcn.py')):	
		print('    deleting', os.path.join(keras_app_path, 'vgg19_fcn.py'))
		os.remove(os.path.join(keras_app_path, 'vgg19_fcn.py'))

	if os.path.isfile(os.path.join(keras_app_path, 'inception_v3_fcn.py')):	
		print('    deleting', os.path.join(keras_app_path, 'inception_v3_fcn.py'))
		os.remove(os.path.join(keras_app_path, 'inception_v3_fcn.py'))

	if os.path.isfile(os.path.join(keras_app_path, 'xception_fcn.py')):	
		print('    deleting', os.path.join(keras_app_path, 'xception_fcn.py'))
		os.remove(os.path.join(keras_app_path, 'xception_fcn.py'))
		
	print('\nSuccessfully uninstall FCN extensions from Keras.')


	





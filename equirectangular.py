import cv2
import numpy as np
import functions as f
import os
import argparse

#Auxiliar functions of equirectangular image
#Main program
def main(scene,common=[1024,512,['lit'],0,1,'cam_rot.txt','R'],specific=[]):

	#----------------------------------------------------------------------------
	#Equirectangular image parameters
	final_w = common[0]	#Image resolution: width
	final_h = common[1]	#Image resolution: height
	mode_list = common[2] #View mode
	init_loc = common[3]	#First location to evaluate
	num_locs = common[4]	#Number of locations
	loc_list = [i + init_loc for i in range(num_locs)] 	#List of locations
	rot1 = common[5]
	rot2 = common[6]
	#----------------------------------------------------------------------------
	if not cv2.useOptimized():
		print('Turn on the Optimizer')
		cv2.setUseOptimized(True)
	#Geometric parameters
	Nor,Rot = f.load_geom()
	

	#Camera images - skybox
	for mode in mode_list:
		print('{} mode composition'.format(mode.capitalize()))
		for loc in loc_list:
			final = np.zeros((final_h,final_w,3), np.uint8)
			r,g,b = np.zeros(final_h*final_w),np.zeros(final_h*final_w),np.zeros(final_h*final_w)

			count = 0
			target_gt = args.dataset_dir
			dira = os.path.join(target_gt,'0_out')   #forward
			dirb = os.path.join(target_gt,'3_out')   #back
			dirc = os.path.join(target_gt,'1_out')   #left
			dird = os.path.join(target_gt,'2_out')   #right
			dire = os.path.join(target_gt,'4_out')   #up
			dirk = os.path.join(target_gt,'5_out')   #down

			a11 = sorted(os.listdir(dira))
			b11 = sorted(os.listdir(dirb))
			c11 = sorted(os.listdir(dirc))
			d11 = sorted(os.listdir(dird))
			e11 = sorted(os.listdir(dire))
			q11 = sorted(os.listdir(dirk))

			for i1, i2, i3, i4, i5, i6 in zip(a11, b11, c11, d11, e11, q11):
				count = count + 1
				n1 = cv2.imread(dira + '/' + i1.format(mode, loc, mode, loc))
				n2 = cv2.imread(dirb + '/' + i2.format(mode, loc, mode, loc))
				n3 = cv2.imread(dirc + '/' + i3.format(mode, loc, mode, loc))
				n4 = cv2.imread(dird + '/' + i4.format(mode, loc, mode, loc))
				n5 = cv2.imread(dire + '/' + i5.format(mode, loc, mode, loc))
				n6 = cv2.imread(dirk + '/' + i6.format(mode, loc, mode, loc))
				save_dir = args.save_dir
				imagenes = [n1, n2, n3, n4, n5, n6]

				im_h, im_w, ch = imagenes[0].shape
				im_w -= 1
				im_h -= 1

				# Camera parameters
				R_cam = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
				R_view = f.camera_rotation(rot1, rot2, loc)  # Rotation matrix of the viewer
				R_world = np.dot(R_view, R_cam)
				FOV = np.pi / 2.0
				fx = (im_w / 2.0) / np.tan(FOV / 2.0)
				fy = (im_h / 2.0) / np.tan(FOV / 2.0)
				K = np.array([[fx, 0, im_w / 2.0], [0, fy, im_h / 2.0], [0, 0, 1]])

				print('making process...')
				# Pixel mapping
				x, y = np.meshgrid(np.arange(final_w), np.arange(final_h))
				theta = (1.0 - 2 * x / float(final_w)) * np.pi
				phi = (0.5 - y / float(final_h)) * np.pi
				ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
				vec = np.array([(cp * ct), (cp * st), (sp)]).reshape(3, final_w * final_h)
				v_abs = np.dot(R_world, vec)
				img_index = f.get_index(v_abs)
				for i in range(img_index.size):
					n, imagen, R = Nor[img_index[i]], imagenes[img_index[i]], Rot[img_index[i]]
					p_x, p_y = f.get_pixel(v_abs[:, i], R, K)
					color = imagen[p_y, p_x]
					r[i], g[i], b[i] = color[0:3]
				final = cv2.merge((r, g, b)).reshape(final_h, final_w, 3)
				
				cv2.imwrite(os.path.join(save_dir, f"{str(count).zfill(6)}.png"), final)


if __name__ =="__main__":
	scene = 'jjun'
	parser = argparse.ArgumentParser(description='Making fisheyeimg.')
	parser.add_argument('--dataset_dir', type=str, default='./dataset')
	parser.add_argument('--save_dir', type=str, default='./dataset/equirectangular')
	parser.add_argument('--Fov', type=int, default=185)
	args = parser.parse_args()
	main(scene)




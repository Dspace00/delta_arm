from uav_project.utils.SimpleMath import tand, sind, cosd
import numpy as np 
# import matplotlib.pyplot as plt 
# import time 
import math 
import torch

class DeltaKinematics: 
	def __init__(self, rod_b = 0.1, rod_ee = 0.2, r_b = 0.074577, r_ee = 0.02495):
		'''
		configs the robot
		rod_B = length of the link connected to the base
		rod_B = length of the link connected to the end-effector
		r_b   = radius of the base			(distance from center to pin joints)
		r_ee  = radius of the end effector 	(distance from center to universal joints)
		'''

		self.rod_b = rod_b
		self.rod_ee = rod_ee
		self.r_b = r_b
		self.r_ee = r_ee
		self.alpha = torch.tensor([0.0, 120.0, 240.0], dtype=torch.float32)

	def fk(self, theta):
		# calculate FK, takes theta(deg)

		rod_b = self.rod_b
		rod_ee = self.rod_ee

		if not isinstance(theta, torch.Tensor):
			theta = torch.tensor(theta, dtype=torch.float32)

		theta1 = theta[0].item()
		theta2 = theta[1].item()
		theta3 = theta[2].item()

		side_ee	= 2/tand(30)*self.r_ee 
		side_b 	= 2/tand(30)*self.r_b

		t = (side_b - side_ee)*tand(30)/2

		y1 = -(t + rod_b*cosd(theta1))
		z1 = -rod_b*sind(theta1)

		y2 = (t + rod_b*cosd(theta2))*sind(30)
		x2 = y2*tand(60)
		z2 = -rod_b*sind(theta2)

		y3 = (t + rod_b*cosd(theta3))*sind(30)
		x3 = -y3*tand(60)
		z3 = -rod_b*sind(theta3)

		dnm = (y2 - y1)*x3 - (y3 - y1)*x2

		w1 = y1**2 + z1**2
		w2 = x2**2 + y2**2 + z2**2
		w3 = x3**2 + y3**2 + z3**2 

		a1 = (z2-z1)*(y3-y1) - (z3-z1)*(y2-y1)
		b1 = -((w2-w1)*(y3-y1) - (w3-w1)*(y2-y1))/2

		a2 = -(z2-z1)*x3 + (z3-z1)*x2
		b2 = ((w2-w1)*x3 - (w3-w1)*x2)/2

		a = a1**2 + a2**2 + dnm**2
		b = 2*(a1*b1 + a2*(b2-y1*dnm) - z1*dnm**2)
		c = (b2 - y1*dnm)**2 + b1**2 + dnm**2*(z1**2 - rod_ee**2)

		d = b**2 - 4*a*c
		if d < 0:
			return -1 

		z0 = -0.5*(b + d**0.5)/a
		x0 = (a1*z0 + b1)/dnm
		y0 = (a2*z0 + b2)/dnm

		# Frame Transform Back
		# The IK uses pos = mat @ _3d_pose, so FK should use _3d_pose = mat.T @ pos
		# pos = torch.tensor([x0, y0, z0], dtype=torch.float32)
		# mat = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
		# _3d_pose = torch.matmul(mat.T, pos)
		# return _3d_pose

		return torch.tensor([x0, y0, z0], dtype=torch.float32)

	def ik(self, _3d_pose):
		# calculates IK, returns theta(deg)		
		# Frame in Delta manipulator is different from the Frame in the UAV. Need to transform.
		rod_ee = self.rod_ee
		rod_b = self.rod_b
		r_ee = self.r_ee
		r_b = self.r_b 
		alpha = self.alpha 

		theta = [0.0, 0.0, 0.0]
        
        # Frame Transform
		mat = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
		if not isinstance(_3d_pose, torch.Tensor):
			_3d_pose = torch.tensor(_3d_pose, dtype=torch.float32)
		
		# Ensure _3d_pose is 1D for dot product or matrix multiplication
		pos = torch.matmul(mat, _3d_pose.view(-1))
		
		x0, y0, z0 = pos[0], pos[1], pos[2]
		
		# Pre-calculate sine and cosine to avoid repetitive tensor operations inside loop
		cos_alpha = torch.cos(torch.deg2rad(alpha))
		sin_alpha = torch.sin(torch.deg2rad(alpha))

		for i in [0, 1, 2]:

			x = x0*cos_alpha[i] + y0*sin_alpha[i]
			y = -x0*sin_alpha[i] + y0*cos_alpha[i]
			z = z0

			# ee_pos = torch.tensor([x, y, z], dtype=torch.float32)

			# E1_pos = ee_pos + torch.tensor([0.0, -r_ee, 0.0], dtype=torch.float32)
			_x0 = x
			_y0 = y - r_ee
			_z0 = z
			_yf = -r_b

			c1 = (_x0**2 + _y0**2 + _z0**2 + rod_b**2 - rod_ee**2 - _yf**2)/(2*_z0)
			c2 = (_yf - _y0)/_z0
			c3 = -(c1 + c2*_yf)**2 + (c2**2+ 1)*rod_b**2

			if c3 < 0:
				# print("non existing point")
				return int(-1)

			J1_y = (_yf - c1*c2 - c3**0.5)/(c2**2 + 1)
			J1_z = c1 + c2*J1_y
			F1_y = -r_b

			val = -J1_z/(F1_y - J1_y)
			theta[i] = torch.atan(torch.tensor(val, dtype=torch.float32)).item()*180/torch.pi
		
		# Return negative theta array as expected by the caller
		return -torch.tensor(theta, dtype=torch.float32)
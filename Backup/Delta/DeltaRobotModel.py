# DeltaRobotModel.py
import numpy as np

class DeltaRobotModel:
    """Delta机械臂运动学模型"""
    
    def __init__(self):
        # 根据PDF表1的参数（转换为米）
        
        self.s_P = 0.02495 * np.sqrt(3)  # 平台等边三角形边长
        self.L = 0.1        # 上臂长度
        self.l = 0.2        # 下臂平行四边形长度
        
        # 其他几何参数（表1）
        self.w_B = 0.074577      # 从{0}到基座近边的平面距离
        self.u_B = 2 * self.w_B  # 从{0}到基座顶点的平面距离   
        self.u_P = 0.02495        # 从{P}到平台顶点的平面距离
        self.w_P = 0.5 * self.u_P # 从{P}到平台近边的平面距离

        # 计算常数a, b, c（PDF第7页）
        self.a = self.w_B - self.u_P
        self.b = 0.5 * self.s_P - (np.sqrt(3)/2) * self.w_B
        self.c = self.w_P - 0.5 * self.w_B
        
        # 基座点坐标（PDF第6页）
        self.B1 = np.array([0, -self.w_B, 0])
        self.B2 = np.array([(np.sqrt(3)/2) * self.w_B, 0.5 * self.w_B, 0])
        self.B3 = np.array([-(np.sqrt(3)/2) * self.w_B, 0.5 * self.w_B, 0])
        
        # 平台点坐标（PDF第6页）
        self.P1 = np.array([0, -self.u_P, 0])
        self.P2 = np.array([0.5 * self.s_P, self.w_P, 0])
        self.P3 = np.array([-0.5 * self.s_P, self.w_P, 0])
        
        # 工作空间限制
        self.workspace_limits = {
            'x': [-0.15, 0.15],
            'y': [-0.15, 0.15], 
            'z': [-0.25, -0.05]
        }
    
    def inverse_kinematics(self, target_pos):
        """
        逆运动学计算 - 基于PDF第8-9页的方法
        输入: 目标位置 [x, y, z] (米)
        输出: 三个关节角度 [theta1, theta2, theta3] (弧度)
        """
        mat = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        pos = np.dot(mat,target_pos).tolist()

        x, y, z = pos
        
        # 计算三个臂的系数（PDF第9页方程）
        E1 = 2 * self.L * (y + self.a)
        F1 = 2 * self.L * z
        G1 = (x**2 + y**2 + z**2 + self.a**2 + self.L**2 + 
              2*y*self.a - self.l**2)
        
        E2 = -self.L * (np.sqrt(3)*(x + self.b) + y + self.c)
        F2 = 2 * self.L * z
        G2 = (x**2 + y**2 + z**2 + self.b**2 + self.c**2 + self.L**2 + 
              2*x*self.b + 2*y*self.c - self.l**2)
        
        E3 = self.L * (np.sqrt(3)*(x - self.b) - y - self.c)
        F3 = 2 * self.L * z
        G3 = (x**2 + y**2 + z**2 + self.b**2 + self.c**2 + self.L**2 - 
              2*x*self.b + 2*y*self.c - self.l**2)
        
        # 使用三角代换求解（PDF第9页）
        def solve_arm(E, F, G):
            """求解单个臂的关节角度"""
            # 使用半角正切代换
            discriminant = F**2 + E**2 - G**2
            
            if discriminant < 0:
                raise ValueError("目标位置不可达")
            
            t1 = (-F + np.sqrt(discriminant)) / (G - E)
            t2 = (-F - np.sqrt(discriminant)) / (G - E)
            
            theta1 = 2 * np.arctan(t1)
            theta2 = 2 * np.arctan(t2)
            
            # 选择肘关节向外的解（knee outside）
            return theta2  # 根据PDF描述选择第二个解
        
        try:
            theta1 = solve_arm(E1, F1, G1)
            theta2 = solve_arm(E2, F2, G2)
            theta3 = solve_arm(E3, F3, G3)
            
            return [-theta2, -theta3, -theta1]
            
        except ValueError as e:
            print(f"逆运动学错误: {e}")
            return None
    
    def forward_kinematics(self, joint_angles):
        """
        正运动学计算 - 基于PDF附录的三球相交算法
        输入: 三个关节角度 [theta1, theta2, theta3] (弧度)
        输出: 末端位置 [x, y, z] (米)
        """
        theta1, theta2, theta3 = joint_angles
        
        # 计算上臂端点（PDF第7页方程2）
        L1 = np.array([0, -self.L * np.cos(theta1), -self.L * np.sin(theta1)])
        L2 = np.array([
            (np.sqrt(3)/2) * self.L * np.cos(theta2),
            0.5 * self.L * np.cos(theta2),
            -self.L * np.sin(theta2)
        ])
        L3 = np.array([
            -(np.sqrt(3)/2) * self.L * np.cos(theta3),
            0.5 * self.L * np.cos(theta3),
            -self.L * np.sin(theta3)
        ])
        
        A1 = self.B1 + L1
        A2 = self.B2 + L2
        A3 = self.B3 + L3
        
        # 计算虚拟球心（PDF第7页）
        A1v = A1 - self.P1
        A2v = A2 - self.P2
        A3v = A3 - self.P3
        
        # 三球相交算法（PDF附录第37-38页）
        x1, y1, z1 = A1v
        x2, y2, z2 = A2v
        x3, y3, z3 = A3v
        
        # 计算系数（PDF第37页）
        a11 = 2 * (x3 - x1)
        a12 = 2 * (y3 - y1)
        a13 = 2 * (z3 - z1)
        b1 = -(x1**2 + y1**2 + z1**2) + (x3**2 + y3**2 + z3**2)
        
        a21 = 2 * (x3 - x2)
        a22 = 2 * (y3 - y2)
        a23 = 2 * (z3 - z2)
        b2 = -(x2**2 + y2**2 + z2**2) + (x3**2 + y3**2 + z3**2)
        
        # 求解z = f(x,y)（PDF第37-38页）
        if abs(a13) < 1e-10 or abs(a23) < 1e-10:
            raise ValueError("奇异位置，无法求解正运动学")
        
        a1 = a11/a13 - a21/a23
        a2 = a12/a13 - a22/a23
        a3 = b2/a23 - b1/a13
        
        a4 = -a2/a1 if abs(a1) > 1e-10 else 0
        a5 = -a3/a1 if abs(a1) > 1e-10 else 0
        
        a6 = (-a21*a4 - a22)/a23
        a7 = (b2 - a21*a5)/a23
        
        # 构建二次方程（PDF第38页）
        A_coeff = a4**2 + 1 + a6**2
        B_coeff = (2*a4*(a5 - x1) - 2*y1 + 2*a6*(a7 - z1))
        C_coeff = (a5*(a5 - 2*x1) + a7*(a7 - 2*z1) + 
                  x1**2 + y1**2 + z1**2 - self.l**2)
        
        # 求解y（PDF第38页方程A.10）
        discriminant = B_coeff**2 - 4*A_coeff*C_coeff
        
        if discriminant < 0:
            raise ValueError("三球无实交点")
        
        y_plus = (-B_coeff + np.sqrt(discriminant)) / (2*A_coeff)
        y_minus = (-B_coeff - np.sqrt(discriminant)) / (2*A_coeff)
        
        # 计算对应的x和z
        x_plus = a4*y_plus + a5
        z_plus = a6*y_plus + a7
        
        x_minus = a4*y_minus + a5
        z_minus = a6*y_minus + a7
        
        # 选择基座下方的解（PDF第8页）
        if z_plus > z_minus:  # z坐标更负的在下方的位置
            return np.array([x_plus, y_plus, z_plus])
        else:
            return np.array([x_minus, y_minus, z_minus])
    
    def check_workspace(self, position):
        """检查位置是否在工作空间内"""
        x, y, z = position
        limits = self.workspace_limits
        
        if (limits['x'][0] <= x <= limits['x'][1] and
            limits['y'][0] <= y <= limits['y'][1] and
            limits['z'][0] <= z <= limits['z'][1]):
            return True
        return False
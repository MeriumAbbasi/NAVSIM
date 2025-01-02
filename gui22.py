from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import all the backend classes
class Traj_char:
    # [Previous Traj_char implementation remains the same]
    def __init__(self, smoothing, coupling, freq, ini_cond):
        self.smoothing = smoothing
        self.coupling = coupling
        self.freq = freq
        self.init_lat = ini_cond[0]*math.pi/180
        self.init_lon = ini_cond[1]*math.pi/180
        self.alt = ini_cond[2]
        self.init_head = ini_cond[3]*math.pi/180
        self.init_pitch = ini_cond[4]*math.pi/180
        self.init_roll = ini_cond[5]*math.pi/180
        self.init_vel = ini_cond[6]

# [Include other backend classes: Trajectory, GPS, IMU as they were]
# Define the Library class here
class Trajectory:
    def __init__(self, start,segment,trajectory):
        self.traj=trajectory

        self.lat0=start[0]*math.pi/180
        self.lon0=start[1]*math.pi/180
        self.height0=start[2]
        self.head0=start[3]*math.pi/180
        self.pitch0=start[4]*math.pi/180
        self.roll0=start[5]*math.pi/180
        self.body_vel0=start[6]*math.pi/180

        self.time_span = segment[0]
        self.head_chg = segment[1]*math.pi/180
        self.pitch_chg = segment[2]*math.pi/180
        self.roll_chg=segment[3]*math.pi/180
        self.body_vel1=segment[4]

        self.head1=self.head0+self.head_chg
        self.pitch1=self.pitch0+self.pitch_chg
        self.roll1=self.roll0+self.roll_chg

        self.Re		= 6378135.072;              #Radius of Earth
        self.e		= 1.0 / 298.3;                  #Eccentricity
        self.g0		= 9.7803267714;                 #Accel due to gravity
        self.WIEE	= 0.00007292115147;             #Earth Spin Rate

    def ComputeCen(self,lamda, fai):
        Cen = np.zeros((3, 3))

        Cen[0][0] = - math.sin(lamda)
        Cen[0][1] = math.cos(lamda)
        Cen[0][2] = 0

        Cen[1][0] = - math.sin(fai) * math.cos(lamda)
        Cen[1][1] = - math.sin(fai) * math.sin(lamda)
        Cen[1][2] = math.cos(fai)

        Cen[2][0] = math.cos(fai) * math.cos(lamda)
        Cen[2][1] = math.cos(fai) * math.sin(lamda)
        Cen[2][2] = math.sin(fai)

        return Cen

    def ComputeTbn(self,psai, theta, gama):
        Tnb=np.zeros((3, 3))
        Tbn=np.zeros((3, 3))

        Tnb[0][0] = math.cos(gama) * math.cos(psai) + math.sin(gama) * math.sin(theta) * math.sin(psai)
        Tnb[0][1] = -math.cos(gama) * math.sin(psai) + math.sin(gama) * math.sin(theta) * math.cos(psai)
        Tnb[0][2] = -math.sin(gama) * math.cos(theta)

        Tnb[1][0] = math.cos(theta) * math.sin(psai)
        Tnb[1][1] = math.cos(theta) * math.cos(psai)
        Tnb[1][2] = math.sin(theta)

        Tnb[2][0] = math.sin(gama) * math.cos(psai) - math.cos(gama) * math.sin(theta) * math.sin(psai)	
        Tnb[2][1] = -math.sin(gama) * math.sin(psai) - math.cos(gama) * math.sin(theta) * math.cos(psai)	
        Tnb[2][2] = math.cos(gama) * math.cos(theta)

        for i in range(3):
            for j in range(3):
                Tbn[i][j] = Tnb[j][i]

        return (Tnb,Tbn)


    def smoothing_coeff(self):
        tspan = self.time_span / 10

        if self.traj.smoothing==1:
             ha0 = self.head0
             ha1 = 0
             ha2 = 3.0 * self.head_chg / self.time_span / self.time_span
             ha3 = -2.0 * self.head_chg / self.time_span / self.time_span / self.time_span

             pa0 = self.pitch0
             pa1 = 0
             pa2 = 3.0 * self.pitch_chg / self.time_span / self.time_span
             pa3 = -2.0 * self.pitch_chg / self.time_span / self.time_span / self.time_span

             ra0 = self.roll0
             ra1 = 0
             ra2 = 3.0 * self.roll_chg / self.time_span / self.time_span
             ra3 = -2.0 * self.roll_chg / self.time_span / self.time_span / self.time_span

             va0 = self.body_vel0
             va1 = 0
             va2 = 3.0 * (self.body_vel1 - self.body_vel0) / self.time_span / self.time_span
             va3 = -2.0 * (self.body_vel1 - self.body_vel0) / self.time_span / self.time_span / self.time_span

             A=[[ha0,ha1,ha2,ha3],
                [pa0,pa1,pa2,pa3],
                [ra0,ra1,ra2,ra3],
                [va0,va1,va2,va3]]

        elif self.traj.smoothing==2:
             hchg = self.head_chg / 20.0
             pchg = self.pitch_chg / 20.0
             rchg = self.roll_chg / 20.0
             vchg = (self.body_vel1 -self.body_vel0) / 20.0

             ha = self.head_chg * 0.9 / self.time_span / 0.8
             pa = self.pitch_chg * 0.9 / self.time_span / 0.8
             ra = self.roll_chg * 0.9 / self.time_span / 0.8
             va = (self.body_vel1 - self.body_vel0) * 0.9 / self.time_span / 0.8

             ha00 = self.head0
             ha10 = 0
             ha20 = (3.0 * hchg - ha * tspan) / tspan / tspan
             ha30 = (-2.0 * hchg + ha * tspan) / tspan / tspan / tspan

             pa00 = self.pitch0
             pa10 = 0
             pa20 = (3.0 * pchg - pa * tspan) / tspan / tspan
             pa30 = (-2.0 * pchg + pa * tspan) / tspan / tspan / tspan

             ra00 = self.roll0
             ra10 = 0
             ra20 = (3.0 * rchg - ra * tspan) / tspan / tspan
             ra30 = (-2.0 * rchg + ra * tspan) / tspan / tspan / tspan

             va00 = self.body_vel0
             va10 = 0
             va20 = (3.0 * vchg  - va * tspan) / tspan / tspan
             va30 = (-2.0 * vchg + va * tspan) / tspan / tspan / tspan

             ha0 = ha
             pa0 = pa
             ra0 = ra
             va0 = va

             ha02 = self.head1 - hchg
             ha12 = ha
             ha22 = -(2 * ha * tspan - 3.0 * hchg) / tspan / tspan
             ha32 = (-2.0 * hchg + ha * tspan) / tspan / tspan / tspan

             pa02 = self.pitch1 - pchg
             pa12 = pa
             pa22 = -(2 * pa * tspan - 3.0 * pchg) / tspan / tspan
             pa32 = (-2.0 * pchg + pa * tspan) / tspan / tspan / tspan

             ra02 = self.roll1 - rchg
             ra12 = ra
             ra22 = -(2 * ra * tspan - 3.0 * rchg) / tspan / tspan
             ra32 = (-2.0 * rchg + ra * tspan) / tspan / tspan / tspan

             va02 = self.body_vel1 - vchg
             va12 = va
             va22 = -(2 * va * tspan - 3.0 * vchg) / tspan / tspan
             va32 = (-2.0 * vchg + va * tspan) / tspan / tspan / tspan

             ha0 = self.head0
             ha1 = 0
             ha2 = 3.0 * self.head_chg / self.time_span / self.time_span
             ha3 = -2.0 * self.head_chg / self.time_span / self.time_span / self.time_span

             pa0 = self.pitch0
             pa1 = 0
             pa2 = 3.0 * self.pitch_chg / self.time_span / self.time_span
             pa3 = -2.0 * self.pitch_chg / self.time_span / self.time_span / self.time_span

             ra0 = self.roll0
             ra1 = 0
             ra2 = 3.0 * self.roll_chg / self.time_span / self.time_span
             ra3 = -2.0 * self.roll_chg / self.time_span / self.time_span / self.time_span

             va0 = self.body_vel0
             va1 = 0
             va2 = 3.0 * (self.body_vel1 - self.body_vel0) / self.time_span / self.time_span
             va3 = -2.0 * (self.body_vel1 - self.body_vel0) / self.time_span / self.time_span / self.time_span

             A=[[ha00,ha10,ha20,ha30,ha02,ha12,ha22,ha32,ha0,ha1,ha2,ha3],
                [pa00,pa10,pa20,pa30,pa02,pa12,pa22,pa32,pa0,pa1,pa2,pa3],
                [ra00,ra10,ra20,ra30,ra02,ra12,ra22,ra32,ra0,ra1,ra2,ra3],
                [va00,va10,va20,va30,va02,va12,va22,va32,va0,va1,va2,va3]]
        
        elif self.traj.smoothing==3:
            ha0 = self.head_chg / self.time_span
            pa0 = self.pitch_chg / self.time_span
            ra0 = self.roll_chg / self.time_span
            va0 = (self.body_vel1 - self.body_vel0) / self.time_span

            A=[ha0,pa0,ra0,va0]

        return A
        
        

    def base_traj(self):
        count = int(self.time_span * self.traj.freq)
        lat     = self.lat0
        lon     = self.lon0
        height  = self.height0

        head    = self.head0
        pitch   = self.pitch0
        roll    = self.roll0

        Cen=self.ComputeCen(lat,lon)
        Tbn=self.ComputeTbn(head,pitch,roll)

        A=self.smoothing_coeff()

        imu_ideal = pd.DataFrame(columns=['Gx', 'Gy','Gz','Ax','Ay','Az'])
        gps_ideal = pd.DataFrame(columns=['Lat', 'Lon','Alt','Vx','Vy','Vz'])
        
        for k in range(count):

            g = self.g0 * (1.0 + 0.00193185138639 * Cen[2][2] * Cen[2][2])/ math.sqrt(1 - 0.00669437999013 * Cen[2][2] * Cen[2][2])* (1.0 - 2.0 * height / self.Re)

            Rm = self.Re * (1.0 - 2.0 * self.e + 3.0 * self.e * Cen[2][2] * Cen[2][2]) + height
            Rn = self.Re * (1.0 + self.e * Cen[2][2] * Cen[2][2]) + height

            delT = k / self.traj.freq
            #print(delT)

            if self.traj.smoothing==1:
                ha0,ha1,ha2,ha3=A[0]
                pa0,pa1,pa2,pa3=A[1]
                ra0,ra1,ra2,ra3=A[2]
                va0,va1,va2,va3=A[3]

                body_vel = va0 + delT * (va1 + delT * (va2 + delT * va3))
                dif_v    = va1 + delT * (2 * va2 + 3 * va3 * delT)

                dif_head    = ha1 + delT * (2 * ha2 + 3 * ha3 * delT)
                dif_pitch   = pa1 + delT * (2 * pa2 + 3 * pa3 * delT)

                head    = ha0 + delT * (ha1 + delT * (ha2 + delT * ha3))
                pitch   = pa0 + delT * (pa1 + delT * (pa2 + delT * pa3))
                

                if (self.head_chg != 0 and self.traj.coupling):
            
                    roll = math.atan(body_vel * dif_head / g)
                    dd_head = 2 * ha2 + 6 * ha3 * delT
                    dif_roll = g * (body_vel * dd_head + dif_head * dif_v) /(g * g + body_vel * body_vel * dif_head * dif_head)
                else:

                    roll = ra0 + delT * (ra1 + delT * (ra2 + delT * ra3))
                    dif_roll    = ra1 + delT * (2 * ra2 + 3 * ra3 * delT)

               

            elif self.traj.smoothing==2:
                ha00,ha10,ha20,ha30,ha02,ha12,ha22,ha32,ha0,ha1,ha2,ha3=A[0]
                pa00,pa10,pa20,pa30,pa02,pa12,pa22,pa32,pa0,pa1,pa2,pa3=A[1]
                ra00,ra10,ra20,ra30,ra02,ra12,ra22,ra32,ra0,ra1,ra2,ra3=A[2]
                va00,va10,va20,va30,va02,va12,va22,va32,va0,va1,va2,va3=A[3]

                if k < self.time_span * self.traj.freq * 0.10:
                    body_vel = va00 + delT * (va10 + delT * (va20 + delT * va30))
                    dif_v    = va10 + delT * (2 * va20 + 3 * va30 * delT)

                    dif_head    = ha10 + delT * (2 * ha20 + 3 * ha30 * delT)
                    dif_pitch   = pa10 + delT * (2 * pa20 + 3 * pa30 * delT)

                    head    = ha00 + delT * (ha10 + delT * (ha20 + delT * ha30))
                    pitch   = pa00 + delT * (pa10 + delT * (pa20 + delT * pa30))
                    
                    print('vel=',body_vel)

                    if (self.head_chg != 0 and self.traj.coupling):
                        roll = math.atan(body_vel * dif_head / g)
                        dd_head = 2 * ha20 + 6 * ha30 * delT
                        dif_roll = g * (body_vel * dd_head + dif_head * dif_v) /(g * g + body_vel * body_vel * dif_head * dif_head)
                
                    else:
                        roll = ra00 + delT * (ra10 + delT * (ra20 + delT * ra30))
                        dif_roll    = ra10 + delT * (2 * ra20 + 3 * ra30 * delT)

                    
                
                elif k >= self.time_span * self.traj.freq * 0.90:
                    TT = delT - self.time_span * 0.90
                    body_vel = va02 + TT * (va12 + TT * (va22 + TT * va32))
                    dif_v    = va12 + TT * (2 * va22 + 3 * va32 * TT)

                    dif_head    = ha12 + TT * (2 * ha22 + 3 * ha32 * TT)
                    dif_pitch   = pa12 + TT * (2 * pa22 + 3 * pa32 * TT)

                    head    = ha02 + TT * (ha12 + TT * (ha22 + TT * ha32))
                    pitch   = pa02 + TT * (pa12 + TT * (pa22 + TT * pa32))

                    print('vel=',body_vel)

                    if (self.head_chg != 0 and self.traj.coupling):
                        roll = math.atan(body_vel * dif_head / g)
                        dd_head = 2 * ha22 + 6 * ha32 * TT
                        dif_roll = g * (body_vel * dd_head + dif_head * dif_v) /(g * g + body_vel * body_vel * dif_head * dif_head)
                    else:
                        roll = ra02 + TT * (ra12 + TT * (ra22 + TT * ra32))
                        dif_roll    = ra12 + TT * (2 * ra22 + 3 * ra32 * TT)

                else:
                     TT = delT - self.time_span * 0.10

                     body_vel = self.body_vel0 + (self.body_vel1 - self.body_vel0) * 0.05 + TT * va0
                     dif_v    = va0

                     dif_head    = ha0
                     dif_pitch   = pa0

                     head    = self.head0 + self.head_chg * 0.05 + TT * ha0
                     pitch   = self.pitch0 + self.pitch_chg * 0.05 + TT * pa0
                     print('vel=',body_vel)

                     if (self.head_chg != 0 and self.traj.coupling):
                        roll = math.atan(body_vel * dif_head / g)
                        dd_head = 0
                        dif_roll = g * (body_vel * dd_head + dif_head * dif_v) /(g * g + body_vel * body_vel * dif_head * dif_head)
                     else:
                        roll = self.roll0 + self.roll_chg * 0.05 + TT * ra0
                        dif_roll    = ra0
          

            elif self.traj.smoothing==3:
                ha0=A[0]
                pa0=A[1]
                ra0=A[2]
                va0=A[3]

                body_vel = self.body_vel0 + delT * va0
                dif_v    = va0

                dif_head    = ha0
                dif_pitch   = pa0

                head    = self.head0 + delT * ha0
                pitch   = self.pitch0 + delT * pa0
                print('vel=',body_vel)

                if (self.head_chg != 0 and self.traj.coupling):
                    roll = math.atan(body_vel * dif_head / g)
                    dd_head = 0
                    dif_roll = g * (body_vel * dd_head + dif_head * dif_v) /(g * g + body_vel * body_vel * dif_head * dif_head)
                else:
                    roll = self.roll0 + delT * ra0
                    dif_roll    = ra0


            (Tnb,Tbn)=self.ComputeTbn(head, pitch, roll)

            Wnbb= [ math.cos(roll) * dif_pitch + math.sin(roll) * math.cos(pitch) * dif_head,
                    dif_roll - math.sin(pitch) * dif_head,
                    math.sin(roll) * dif_pitch - math.cos(pitch) * math.cos(roll) * dif_head]
            
            vx = Tbn[0][1] * body_vel
            vy = Tbn[1][1] * body_vel
            vz = Tbn[2][1] * body_vel

            dif_vx  = -dif_pitch * math.sin(pitch) * math.sin(head) * body_vel+ dif_head * math.cos(pitch) * math.cos(head) * body_vel + dif_v * math.cos(pitch) * math.sin(head)
            dif_vy  = -dif_pitch * math.sin(pitch) * math.cos(head) * body_vel- dif_head * math.cos(pitch) * math.sin(head) * body_vel+ dif_v * math.cos(pitch) * math.cos(head)
            dif_vz  = dif_pitch * math.cos(pitch) * body_vel+ dif_v * math.sin(pitch)

            lat     += vy / self.traj.freq / Rm
            lon     += vx / self.traj.freq/ Rn / math.cos(lat)
            height  += vz / self.traj.freq

            Wien= np.array([self.WIEE * Cen[0][2],
    	                    self.WIEE * Cen[1][2],
	                        self.WIEE * Cen[2][2]])
            
            Wenn= np.array([-vy / Rm,
    	                    vx / Rn,
	                        vx * math.tan(lat) / Rn])
            
            Winn= Wien + Wenn

            Winb =np.dot(Tnb , Winn)

            Wibb = Wnbb + Winb

            fbn= np.array([dif_vx - (2 * Wien[2] + Wenn[2]) * vy + (2 * Wien[1] + Wenn[1]) * vz,
                           dif_vy + (2 * Wien[2] + Wenn[2]) * vx - (2 * Wien[0] + Wenn[0]) * vz,
                           dif_vz - (2 * Wien[1] + Wenn[1]) * vx+ (2 * Wien[0] + Wenn[0]) * vy + g])
            
            fb = np.dot(Tnb,fbn) 

            Cen=self.ComputeCen(lat, lon)

            imu = {'Gx':  Wibb[0], 'Gy': Wibb[1], 'Gz': Wibb[2],'Ax':fb[0],'Ay':fb[1],'Az':fb[2]}
            gps = {'Lat': lat*180/math.pi, 'Lon': lon*180/math.pi,'Alt': height,'Vx': vx,'Vy': vy,'Vz': vz}

            imu_ideal = pd.concat([imu_ideal, pd.DataFrame([imu])], ignore_index=True) 
            gps_ideal = pd.concat([gps_ideal, pd.DataFrame([gps])], ignore_index=True) 


        next_start=[lat*180/math.pi,lon*180/math.pi,height,head*180/math.pi,pitch*180/math.pi,roll*180/math.pi,body_vel]

                
        return gps_ideal,imu_ideal,next_start
       
# GPS Class
class GPS:
    def __init__(self, GPS_errors,base_gps):
        self.Vel_err = GPS_errors[0]
        self.Pos_err = (GPS_errors[1]/30/3600)

        self.vx=base_gps['Vx']
        self.vy=base_gps['Vy']
        self.vz=base_gps['Vz']

        self.Lat=base_gps['Lat']
        self.Lon=base_gps['Lon']
        self.Alt=base_gps['Alt']


    def Add_errors(self):
        #Add Errors
        Vx_err=self.vx + random.gauss(0, self.Vel_err)
        Vy_err=self.vy + random.gauss(0, self.Vel_err)
        Vz_err=self.vz + random.gauss(0, self.Vel_err)

        Lat_err=self.Lat + random.gauss(0, self.Pos_err)
        Lon_err=self.Lon + random.gauss(0, self.Pos_err)
        Alt_err=self.Alt + random.gauss(0, self.Pos_err)*30*3600

        real_gps=pd.DataFrame([Lat_err,Lon_err,Alt_err,Vx_err,Vy_err, Vz_err])

        return real_gps
    
# IMU Class
class IMU:
    def __init__(self, accel_errors, gyro_errors, base_imu):
        # Accel Errors
        self.accel_Constant_bias = accel_errors['CB']
        self.accel_Random_bias = accel_errors['RB']
        self.accel_SFx = accel_errors['SFx'] / 100
        self.accel_SFy = accel_errors['SFy'] / 100
        self.accel_SFz = accel_errors['SFz'] / 100
        self.accel_Kxy = accel_errors['Kxy']
        self.accel_Kxz = accel_errors['Kxz']
        self.accel_Kyx = accel_errors['Kyx']
        self.accel_Kyz = accel_errors['Kyz']
        self.accel_Kzx = accel_errors['Kzx']
        self.accel_Kzy = accel_errors['Kzy']
        self.LAx = accel_errors['Lx']
        self.LAy = accel_errors['Ly']
        self.LAz = accel_errors['Lz']

        # Gyro Errors
        self.gyro_Constant_bias = gyro_errors['CB']
        self.gyro_Random_bias = gyro_errors['RB']
        self.gyro_SFx = gyro_errors['SFx'] / 100
        self.gyro_SFy = gyro_errors['SFy'] / 100
        self.gyro_SFz = gyro_errors['SFz'] / 100
        self.gyro_Kxy = gyro_errors['Kxy']
        self.gyro_Kxz = gyro_errors['Kxz']
        self.gyro_Kyx = gyro_errors['Kyx']
        self.gyro_Kyz = gyro_errors['Kyz']
        self.gyro_Kzx = gyro_errors['Kzx']
        self.gyro_Kzy = gyro_errors['Kzy']

        # Base IMU
        self.Gx = base_imu['Gx']
        self.Gy = base_imu['Gy']
        self.Gz = base_imu['Gz']

        self.Ax = base_imu['Ax']
        self.Ay = base_imu['Ay']
        self.Az = base_imu['Az']

    def Add_errors(self):
        Gx_err = (
            self.Gx + self.gyro_Constant_bias + random.gauss(0, self.gyro_Random_bias)
            + self.gyro_SFx * self.Gx + self.gyro_Kxy * self.Gy + self.gyro_Kxz * self.Gz
        )

        Gy_err = (
            self.Gy + self.gyro_Constant_bias + random.gauss(0, self.gyro_Random_bias)
            + self.gyro_SFy * self.Gy + self.gyro_Kyx * self.Gx + self.gyro_Kyz * self.Gz
        )

        Gz_err = (
            self.Gz + self.gyro_Constant_bias + random.gauss(0, self.gyro_Random_bias)
            + self.gyro_SFz * self.Gz + self.gyro_Kzx * self.Gx + self.gyro_Kzy * self.Gy
        )

        Ax_err = (
            self.Ax + self.accel_Constant_bias + random.gauss(0, self.accel_Random_bias)
            + self.accel_SFx * self.Ax + self.accel_Kxy * self.Ay + self.accel_Kxz * self.Az + self.Gx * self.LAx
        )

        Ay_err = (
            self.Ay + self.accel_Constant_bias + random.gauss(0, self.accel_Random_bias)
            + self.accel_SFy * self.Ay + self.accel_Kyx * self.Ax + self.accel_Kyz * self.Az + self.Gy * self.LAy
        )

        Az_err = (
            self.Az + self.accel_Constant_bias + random.gauss(0, self.accel_Random_bias)
            + self.accel_SFz * self.Az + self.accel_Kzx * self.Ax + self.accel_Kzy * self.Ay + self.Gz * self.LAz
        )


        real_imu = pd.DataFrame([Gx_err,Gy_err,Gz_err,Ax_err,Ay_err,Az_err])

        return real_imu
        

class NavSimGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # [Previous GUI initialization code remains the same]
        self.setWindowTitle("NavSim - Navigation Sensor Simulator") 
        self.setGeometry(100, 100, 1250, 750)

         # Set window icon
        window_icon = QIcon("C:/Users/Admin/Downloads/GUI_final/navsim_icon.png")
        self.setWindowIcon(window_icon)

        self.add_icon = QIcon("C:/Users/Admin/Downloads/GUI_final/add.png")
        self.remove_icon = QIcon("C:/Users/Admin/Downloads/GUI_final/trash.png")
        self.done_icon = QIcon("C:/Users/Admin/Downloads/GUI_final/check.png")


        self.setStyleSheet("""
            QMainWindow {
                background-color: #14213d;
            }
            QLabel {
                color: #e5e5e5;
                font-size: 14px;
            }
            QGroupBox {
                background-color: #293241;
                border: 2px solid #3d5a80;
                border-radius: 8px;
                margin-top: 10px;
                font-size: 16px;
                color: #e5e5e5;
            }
            QLineEdit {
                background-color: #3d5a80;
                color: white;
                border: 1px solid #98c1d9;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton {
                background-color: #98c1d9;
                color: #293241;
                border-radius: 4px;
                padding: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3d5a80;
                color: white;
            }
            QTabWidget::pane {
                border: 2px solid #3d5a80;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #293241;
                color: #e5e5e5;
                padding: 8px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3d5a80;
            }
            QTreeView {
                background-color: #293241;
                color: #e5e5e5;
                border: 2px solid #3d5a80;
                border-radius: 4px;
            }
            QTreeView::item:selected {
                background-color: #3d5a80;
            }
            QComboBox {
                background-color: #3d5a80;
                color: white;
                border: 1px solid #98c1d9;
                border-radius: 4px;
                padding: 4px;
            }
        """)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Create tabs
        self.initial_conditions_tab = QWidget()
        self.trajectory_tab = QWidget()
        
        self.tab_widget.addTab(self.initial_conditions_tab, "Initial Conditions")
        self.tab_widget.addTab(self.trajectory_tab, "Trajectory")

        self.setup_initial_conditions_tab()
        self.setup_trajectory_tab()
        
        # Add new method connections
        self.trajectory_segments = []
        self.current_data = None

        # Initialize the Matplotlib figure and canvas
        #self.figure = Figure()
        #self.canvas = FigureCanvas(self.figure)
        
        # Add the canvas to the GUI
        #graph_group = QGroupBox("Graph")
        #graph_layout = QVBoxLayout(graph_group)
        #graph_layout.addWidget(self.canvas)
        
        # Add the graph group to the trajectory tab
        #self.trajectory_tab.layout().addWidget(graph_group)


    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.dir_edit.setText(directory)


    def setup_initial_conditions_tab(self):
       # Main layout
        main_layout = QHBoxLayout(self.initial_conditions_tab)


        
        
        # IMU Parameters with scroll
        imu_scroll = QScrollArea()
        imu_scroll.setWidgetResizable(True)
        imu_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        imu_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            # IMU Parameters
        imu_group = QGroupBox("IMU Parameters")
        imu_layout = QFormLayout()
        imu_group.setLayout(imu_layout)
        imu_scroll.setWidget(imu_group)
        imu_scroll.setFixedWidth(400)


  
        # Add IMU fields
        self.imu_freq = QLineEdit()
        imu_layout.addRow("IMU Frequency (Hz):", self.imu_freq)

        # Add Accelerometer section
            # Accelerometer section
        accel_label = QLabel("Accelerometers")
        accel_label.setStyleSheet("font-weight: bold; color: #98c1d9;")
        imu_layout.addRow(accel_label)

        # Accelerometer Bias
        self.accel_cb = QLineEdit()
        imu_layout.addRow("Constant Bias (g):", self.accel_cb)
        
        self.accel_rb = QLineEdit()
        imu_layout.addRow("Random Bias (g):", self.accel_rb)

        # Scale Factor Errors
        self.accel_sfx = QLineEdit()
        imu_layout.addRow("Ax Scale Factor Error (%):", self.accel_sfx)
        
        self.accel_sfy = QLineEdit()
        imu_layout.addRow("Ay Scale Factor Error (%):", self.accel_sfy)
        
        self.accel_sfz = QLineEdit()
        imu_layout.addRow("Az Scale Factor Error (%):", self.accel_sfz)

        # Misalignments
        self.accel_kxy = QLineEdit()
        imu_layout.addRow("Misalignment Kxy (arcmin):", self.accel_kxy)
        
        self.accel_kxz = QLineEdit()
        imu_layout.addRow("Misalignment Kxz (arcmin):", self.accel_kxz)
        
        self.accel_kyx = QLineEdit()
        imu_layout.addRow("Misalignment Kyx (arcmin):", self.accel_kyx)
        
        self.accel_kyz = QLineEdit()
        imu_layout.addRow("Misalignment Kyz (arcmin):", self.accel_kyz)
        
        self.accel_kzx = QLineEdit()
        imu_layout.addRow("Misalignment Kzx (arcmin):", self.accel_kzx)
        
        self.accel_kzy = QLineEdit()
        imu_layout.addRow("Misalignment Kzy (arcmin):", self.accel_kzy)

        # Lever Arms
        self.accel_lx = QLineEdit()
        imu_layout.addRow("X-axis Lever Arm (m):", self.accel_lx)
        
        self.accel_ly = QLineEdit()
        imu_layout.addRow("Y-axis Lever Arm (m):", self.accel_ly)
        
        self.accel_lz = QLineEdit()
        imu_layout.addRow("Z-axis Lever Arm (m):", self.accel_lz)

        # Store accelerometer parameters in dictionary
        self.Accel_Err = {
            'CB': self.accel_cb,
            'RB': self.accel_rb,
            'SF': [self.accel_sfx, self.accel_sfy, self.accel_sfz],
            'MA': [self.accel_kxy, self.accel_kxz, self.accel_kyx, 
                self.accel_kyz, self.accel_kzx, self.accel_kzy],
            'LA': [self.accel_lx, self.accel_ly, self.accel_lz]
    }
        # Add Gyroscope section
        gyro_label = QLabel("Gyroscopes")
        gyro_label.setStyleSheet("font-weight: bold; color: #98c1d9;")
        imu_layout.addRow(gyro_label)

        # Gyro Bias
        self.gyro_cb = QLineEdit()
        imu_layout.addRow("Constant Bias (°/h):", self.gyro_cb)
        
        self.gyro_rb = QLineEdit()
        imu_layout.addRow("Random Bias (°/h):", self.gyro_rb)

        # Gyro Scale Factors
        self.gyro_sfx = QLineEdit()
        imu_layout.addRow("Gx Scale Factor Error (%):", self.gyro_sfx)
        
        self.gyro_sfy = QLineEdit()
        imu_layout.addRow("Gy Scale Factor Error (%):", self.gyro_sfy)
        
        self.gyro_sfz = QLineEdit()
        imu_layout.addRow("Gz Scale Factor Error (%):", self.gyro_sfz)

        # Gyro Misalignments
        self.gyro_kxy = QLineEdit()
        imu_layout.addRow("Misalignment Kxy (arcmin):", self.gyro_kxy)
        
        self.gyro_kxz = QLineEdit()
        imu_layout.addRow("Misalignment Kxz (arcmin):", self.gyro_kxz)
        
        self.gyro_kyx = QLineEdit()
        imu_layout.addRow("Misalignment Kyx (arcmin):", self.gyro_kyx)
        
        self.gyro_kyz = QLineEdit()
        imu_layout.addRow("Misalignment Kyz (arcmin):", self.gyro_kyz)
        
        self.gyro_kzx = QLineEdit()
        imu_layout.addRow("Misalignment Kzx (arcmin):", self.gyro_kzx)
        
        self.gyro_kzy = QLineEdit()
        imu_layout.addRow("Misalignment Kzy (arcmin):", self.gyro_kzy)
        self.Gyro_Err = {
                'CB': self.gyro_cb,
                'RB': self.gyro_rb, 
                'SF': [self.gyro_sfx, self.gyro_sfy, self.gyro_sfz],
                'MA': [self.gyro_kxy, self.gyro_kxz, self.gyro_kyx,
                    self.gyro_kyz, self.gyro_kzx, self.gyro_kzy]
            }


   
   
        
        # GPS Parameters


    

        #GPS Parameters
        gps_group = QGroupBox("GPS Parameters")
        gps_layout = QFormLayout()
        gps_group.setLayout(gps_layout)

        self.gps_freq = QLineEdit()
        gps_layout.addRow("GPS Frequency (Hz):", self.gps_freq)

        self.gps_vel = QLineEdit()
        gps_layout.addRow("Velocity Covariance (m/s):", self.gps_vel)

        self.gps_pos = QLineEdit()
        gps_layout.addRow("Position Covariance (m):", self.gps_pos)


        # Initial Conditions
        init_group = QGroupBox("Initial Conditions")
        init_layout = QFormLayout()
        init_group.setLayout(init_layout)

        # Position parameters
        self.lat = QLineEdit()
        init_layout.addRow("Initial Latitude (°):", self.lat)

        self.lon = QLineEdit()
        init_layout.addRow("Initial Longitude (°):", self.lon)

        self.alt = QLineEdit()
        init_layout.addRow("Initial Altitude (m):", self.alt)

        self.heading = QLineEdit()
        init_layout.addRow("Initial Heading (°):", self.heading)

        self.pitch = QLineEdit()
        init_layout.addRow("Initial Pitch (°):", self.pitch)

        self.roll = QLineEdit()
        init_layout.addRow("Initial Roll (°):", self.roll)

        self.velocity = QLineEdit()
        init_layout.addRow("Initial Velocity (m/s):", self.velocity)

        # Smoothing options
        smooth_label = QLabel("Smoothing Type")
        smooth_label.setStyleSheet("font-weight: bold; color: #98c1d9;")
        init_layout.addRow(smooth_label)

        self.smooth_group = QButtonGroup()
        radio_style = """
                QRadioButton {
                    color: #e5e5e5;
                    padding: 2px;
                }
                QRadioButton:hover {
                  color: #00adb5;
            }
        """
        smooth3 = QRadioButton("3rd Order Smooth")
        smooth3.setStyleSheet(radio_style)

        smooth10 = QRadioButton("10% 3rd Order Smooth")
        smooth10.setStyleSheet(radio_style)

        smooth_uni = QRadioButton("Uniformly Accelerated")
        smooth_uni.setStyleSheet(radio_style)

        init_layout.addRow(smooth3)
        init_layout.addRow(smooth10)
        init_layout.addRow(smooth_uni)
        self.smooth_group.addButton(smooth3)
        self.smooth_group.addButton(smooth10)
        self.smooth_group.addButton(smooth_uni)

        # Add coupling checkbox
        coupling_label = QLabel("Coupling")
        coupling_label.setStyleSheet("font-weight: bold; color: #98c1d9;")
        init_layout.addRow(coupling_label)
        checkbox_style = """
            QCheckBox {
                color: #e5e5e5;
                padding: 2px;
            }
            QCheckBox:hover {
                color: #00adb5;
            }
            QCheckBox:checked {
                color: #00adb5;
            }
        """
        self.yaw_roll = QCheckBox("Yaw Roll Coupling")
        self.yaw_roll.setStyleSheet(checkbox_style)
        init_layout.addRow(self.yaw_roll)


        dir_label = QLabel("Directory for Output Files")
        dir_label.setStyleSheet("font-weight: bold; color: #98c1d9;")
        init_layout.addRow(dir_label)

        dir_widget = QWidget()
        dir_layout = QHBoxLayout(dir_widget)
        dir_layout.setContentsMargins(0, 0, 0, 0)

        self.dir_edit = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #00adb5;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #393e46;
            }
        """)
        browse_btn.clicked.connect(self.browse_directory)

        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(browse_btn)
        init_layout.addRow(dir_widget)
        # Add all groups to main layout
        main_layout.addWidget(imu_scroll)
        main_layout.addWidget(gps_group)
        main_layout.addWidget(init_group)

    def setup_trajectory_tab(self):
        main_layout = QVBoxLayout(self.trajectory_tab)
        main_layout.setSpacing(5)

        # up side - Table section
        table_container = QWidget()
        table_container.setFixedHeight(300)  # Restrict table height
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(5, 5, 5, 0)  # Reduce margins
        table_layout.setSpacing(5)
        
        # Scroll area for table
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Container widget for table
        container = QWidget()
        container_layout = QVBoxLayout(container)

        # Trajectory table with grid styling
        self.table = QTreeWidget()
        self.table.setHeaderLabels(["Segment", "Duration (s)", "Heading Change (°)", 
                                "Pitch Change (°)", "Roll Change (°)", "Final Velocity (m/s)"])
        self.table.setColumnCount(6)
        
        
        # Grid-like styling
        font = QFont()
        font.setPointSize(12)
        self.table.setFont(font)
            
            # Set row height
        self.table.setStyleSheet("""
                QTreeWidget {
                    background-color: #293241;
                    border: 1px solid #3d5a80;
                    gridline-color: #3d5a80;
                    selection-background-color: #00adb5;
                }
                QTreeWidget::item {
                    border-bottom: 1px solid #3d5a80;
                    border-right: 1px solid #3d5a80;
                    padding: 8px;
                    color: #e5e5e5;
                    min-height: 30px;
                }
                QHeaderView::section {
                    background-color: #14213d;
                    color: #e5e5e5;
                    border: 1px solid #3d5a80;
                    padding: 8px;
                    font-weight: bold;
                    font-size: 12pt;
                }
                QTreeWidget::item:selected {
                    background-color: #00adb5;
                    color: white;
                }
                QLineEdit {
                    color: #293241;
                    background: white;
                    selection-background-color: #00adb5;
                    font-size: 12pt;
                    padding: 5px;
                    min-height: 25px;
                }
        """)

            # Adjust column widths for larger text
        for i in range(6):
            self.table.setColumnWidth(i, 180)

        table_layout.addWidget(self.table)


        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(5, 0, 5, 5)  # Minimize top margin
        button_layout.setSpacing(5)
        # Create buttons with icons
        add_btn = QPushButton()
        add_btn.setIcon(self.add_icon)
        add_btn.setText("Add")
        
        remove_btn = QPushButton()
        remove_btn.setIcon(self.remove_icon)
        remove_btn.setText("Remove")
        
        done_btn = QPushButton()
        done_btn.setIcon(self.done_icon)
        done_btn.setText("Done")
        
        # Style buttons
        button_style = """
            QPushButton {
                background-color: #008ba3;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #393e46;
            }
        """
        
        add_btn.setStyleSheet(button_style)
        remove_btn.setStyleSheet(button_style)
        done_btn.setStyleSheet(button_style)
        
        # Connect signals
        add_btn.clicked.connect(self.add_row)
        remove_btn.clicked.connect(self.remove_row)
        done_btn.clicked.connect(self.generate_trajectory)
        
        # Add to layout
        button_layout.addWidget(add_btn)
        button_layout.addWidget(remove_btn)
        button_layout.addWidget(done_btn)
       
        table_layout.addWidget(self.table)
        table_layout.addLayout(button_layout)

        # Graph section
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        graph_group = QGroupBox("Graph")
        graph_layout = QVBoxLayout(graph_group)
        graph_layout.addWidget(self.canvas)

        # Add both sections to main layout
        main_layout.addWidget(table_container)
        main_layout.addWidget(graph_group)
        
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory()
        if directory:
            self.dir_edit.setText(directory)

    def add_row(self):
        item = QTreeWidgetItem(self.table)
        item.setText(0, str(self.table.topLevelItemCount()))
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        
    def remove_row(self):
        current_item = self.table.currentItem()
        if current_item:
            self.table.takeTopLevelItem(self.table.indexOfTopLevelItem(current_item))

    def finalize_trajectory(self):
        print("Trajectory finalized")

        
    def generate_trajectory(self):
        #try:
            # Get initial conditions from GUI
            ini_cond = [
                float(self.lat.text()),
                float(self.lon.text()),
                float(self.alt.text()),
                float(self.heading.text()),
                float(self.pitch.text()),
                float(self.roll.text()),
                float(self.velocity.text())
            ]
            
            # Determine smoothing type
            smoothing_type = 1  # default
            for button in self.smooth_group.buttons():
                if button.isChecked():
                    if "3rd Order" in button.text():
                        smoothing_type = 1
                    elif "10%" in button.text():
                        smoothing_type = 2
                    else:
                        smoothing_type = 3
            
            # Create trajectory characteristics
            freq = float(self.imu_freq.text())
            coupling = self.yaw_roll.isChecked()
            traj_char = Traj_char(smoothing_type, coupling, freq, ini_cond)
            
            # Process each trajectory segment
            all_gps_data = []
            all_imu_data = []
            current_start = ini_cond
            
            root = self.table.invisibleRootItem()
            for i in range(root.childCount()):
                item = root.child(i)
                segment = [
                    float(item.text(1)),  # duration
                    float(item.text(2)),  # heading change
                    float(item.text(3)),  # pitch change
                    float(item.text(4)),  # roll change
                    float(item.text(5))   # final velocity
                ]


                print(current_start)
                print(segment)
                

                # Generate trajectory for segment
                traj = Trajectory(current_start, segment, traj_char)
                gps_data, imu_data, next_start = traj.base_traj()
                
                
                all_gps_data.append(gps_data)
                all_imu_data.append(imu_data)
                current_start = next_start

                
            
            # Concatenate all segments
            final_gps = pd.concat(all_gps_data, ignore_index=True)
            final_imu = pd.concat(all_imu_data, ignore_index=True)
            
            # Save data
            output_dir = self.dir_edit.text()
            if output_dir:
                gps_file = os.path.join(output_dir, 'GPS_data.txt')
                imu_file = os.path.join(output_dir, 'IMU_data.txt')
                
                final_gps.to_csv(gps_file, index=False, sep='\t')
                final_imu.to_csv(imu_file, index=False, sep='\t')
                
            # Store data for plotting
            self.current_data = final_gps
            self.update_plot()
            
            QMessageBox.information(self, "Success", "Trajectory generated successfully!")
            
        #except Exception as e:
           # QMessageBox.critical(self, "Error", f"Failed to generate trajectory: {str(e)}")
    
    def update_plot(self):
        #if self.current_data is None:
           # return

        gps_data = self.current_data
        #plot_type = self.plot_combo.currentText()
        print(gps_data['Lon'])

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(gps_data['Lon'],gps_data['Lat'],label="Lon-Lat")

        
        ax.legend()
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        self.canvas.draw()

    #def finalize_trajectory(self):
     #   self.generate_trajectory()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NavSimGUI()
    window.show()
    sys.exit(app.exec_())
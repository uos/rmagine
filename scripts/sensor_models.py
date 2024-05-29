import numpy as np


"""
This file provides a translation of rmagine sensor models for test/visualization purposes
"""


def deg2rad(deg):
    return deg * np.pi / 180.0

def rad2deg(rad):
    return rad * 180.0 / np.pi

class SphericalModel:
    # PHI: vertical, y-rot, pitch, polar angle, height
    phi_min = deg2rad(-45.0)
    phi_size = 32
    phi_inc = deg2rad(90.0 / 32)

    # THETA: horizontal, z-rot, yaw, azimuth, width
    theta_min = -np.pi
    theta_size = 1024
    theta_inc = 2.0 * np.pi / 1024

    # RANGE: 
    range_min = 0.5
    range_max = 80.0

    def __init__(self):
        pass

    def getHeight(self):
        return self.phi_size
    
    def getWidth(self):
        return self.theta_size

    def getPhi(self, phi_id):
        return self.phi_min + phi_id * self.phi_inc

    def getTheta(self, theta_id):
        return self.theta_min + theta_id * self.theta_inc

    def getDirection(self, phi_id, theta_id):
        phi = self.getPhi(phi_id)
        theta = self.getTheta(theta_id)
        return np.array([
            np.cos(phi) * np.cos(theta), 
            np.cos(phi) * np.sin(theta), 
            np.sin(phi)])
    

class PinholeModel:
    width = 1
    height = 1

    range_min = 0.5
    range_max = 100.0

    # focal length fx and fy
    f = [0.0, 0.0]
    # center cx and cy
    c = [0.0, 0.0]

    def __init__(self):
        pass

    def getWidth(self):
        return self.width
    
    def getHeight(self):
        return self.height
    
    def getDirectionOptical(self, vid, hid):
        pX = (hid - self.c[0]) / self.f[0]
        pY = (vid - self.c[1]) / self.f[1]
        dir_optical = np.array([pX, pY, 1.0])
        dir_optical /= np.linalg.norm(dir_optical)
        return dir_optical / np.linalg.norm(dir_optical)

    def getDirection(self, vid, hid):
        dir_optical = self.getDirectionOptical(vid, hid)
        dir_normal = np.array([dir_optical[2], -dir_optical[0], -dir_optical[1]])
        return dir_normal
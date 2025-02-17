#Low-cost, open-access refractive index mapping of algal cells using the transport of intensity equation
#Paper available here: (https://www.biorxiv.org/content/10.1101/640755v1)
#Transport of Intesity Equation code
#Adapted from code availabe with 'Transport of Intensity Equation Microscopy for Dynamic Microtubules' by Q. Tyrell Davis
#Davis paper available here:(https://arxiv.org/abs/1707.04139)


#Contents of this code:
#Part1 - Import neccessary libraries
#Part2 - Set up initial parameters
#Part3 - Transport of intensity equation adapted from Davis
#Part4 - Refractive index extraction
#Part5 - Display TIE and RI images, save images, and save csv files.



#Part1 - Import plotting essentials and necessary numpy/scipy #

from ast import Lambda
from fiji.util.gui import GenericDialogPlus
from ij import IJ, ImagePlus
from ij.gui import Roi, GenericDialog
from ij.io import OpenDialog
from ij.process import FloatProcessor
from ij import WindowManager as wm

from ij.plugin import Stack_Statistics, ImageCalculator
from ij.plugin.frame import RoiManager

from loci.plugins import BF

import array
import math
import time
import sys

from org.bytedeco.javacpp.opencv_core import dft, idft, DFT_COMPLEX_OUTPUT, DFT_REAL_OUTPUT, DFT_INVERSE
from org.bytedeco.javacpp.opencv_core import Mat
from ijopencv.ij      import ImagePlusMatConverter
from ijopencv.opencv  import MatImagePlusConverter

progress = 0

def fast_dst(x):  
    n = len(x)
    y = [0] * n
    input0 = [0] + x + [0] * (n + 1)
    
    imp2mat = ImagePlusMatConverter()
    mat2ip = MatImagePlusConverter()
    
    pixel_matrix = split_list(input0, wanted_parts = 1)
    pixel_matrix = [list(x) for x in zip(*pixel_matrix)]
    img = ImagePlus("FFT", FloatProcessor(pixel_matrix))
    
    ImMat = imp2mat.toMat(img.getProcessor())
    ImMat_out = Mat()
    dft(ImMat, ImMat_out, DFT_COMPLEX_OUTPUT, 0)
    
    NewIP  = mat2ip.toImageProcessor(ImMat_out.reshape(1))
    ret = NewIP.getPixels()
    
    for i in range(n):
        y[i] = - ret[2 * i + 3]  # Take Imaginaries
    
    return y


def fft_poisson(dI, h, Dimx, Dimy):  # The imput dI is a Dimy x Dimx matrix.
    b_bar = [0] * (Dimx * Dimy)
    u_bar = [0] * (Dimx * Dimy)
    u = [0] * (Dimx * Dimy)
    lambda_x = [0] * (Dimx)
    lambda_y = [0] * (Dimy)
    #mat = [0] * (2*Dim*Dim + 2)
    
    temp_x = [0] * Dimx
    temp_y = [0] * Dimy
    
    global progress
    step = 1.0/(4*float(Dimx + Dimy))
    
    #n = Dim * Dim
    #base = 2 * math.pi / (2 * n + 2)
    #base0 = math.e ** (- base * 1j)
    #for k in range(2*n + 2):
    #        mat[k] = base0 ** (k)
    
    print("Step1")
    
    
    for i in range(Dimx):
        for j in range(Dimy):
            temp_y[j] = dI[j * Dimx + i]
        ret = fast_dst(temp_y)
        for j in range(Dimy):
            b_bar[j * Dimx + i] = ret[j]
            
        progress = progress + step
        IJ.showProgress(progress)
    
    for i in range(Dimy):
        for j in range(Dimx):
            temp_x[j] = b_bar[i * Dimx + j]
        ret = fast_dst(temp_x)
        for j in range(Dimx):
            b_bar[i * Dimx + j] = ret[j]
            
        progress = progress + step
        IJ.showProgress(progress)
            
    
    temp1 = 2.0/math.sqrt( float(Dimx * Dimy) )
    for i in range(Dimx * Dimy):
        b_bar[i] = b_bar[i] * temp1
        
    for i in range(Dimx):
        lambda_x[i] = -4 * math.sin( (i + 1) * math.pi / (2 * Dimx) ) ** 2
    for i in range(Dimy):
        lambda_y[i] = -4 * math.sin( (i + 1) * math.pi / (2 * Dimy) ) ** 2
        
    print("Step2")

    h2 = h ** 2
    for i in range(Dimy):
        for j in range(Dimx):
            u_bar[i * Dimx + j] = ( h2 * b_bar[i * Dimx + j] ) / (lambda_y[i] + lambda_x[j])
        

    for i in range(Dimx):
        for j in range(Dimy):
            temp_y[j] = u_bar[j * Dimx + i]
        ret = fast_dst(temp_y)
        for j in range(Dimy):
            u[j * Dimx + i] = ret[j]
            
        progress = progress + step
        IJ.showProgress(progress)
    
    for i in range(Dimy):
        for j in range(Dimx):
            temp_x[j] = u[i * Dimx + j]
        ret = fast_dst(temp_x)
        for j in range(Dimx):
            u[i * Dimx + j] = ret[j]
            
        progress = progress + step
        IJ.showProgress(progress)

    
    for i in range(Dimx * Dimy):
        u[i] = u[i] * temp1
        
    return u
    


def high2d(arr, Dim, r):
    s = len(arr)
    fr = [0] * (Dim * Dim)
    fr_shifted = [0] * (Dim * Dim)
    arr1 = [0] * (Dim * Dim)
    temp = [0] * (Dim)
    r2 = float(2 * r ** 2)
    
    imp2mat = ImagePlusMatConverter()
    mat2ip = MatImagePlusConverter()

    print("Start Drawing Graph")
    
    # FFT2
    
    pixel_matrix = split_list(arr, wanted_parts = Dim)
    pixel_matrix = [list(x) for x in zip(*pixel_matrix)]
    img = ImagePlus("FFT", FloatProcessor(pixel_matrix))
    
    ImMat = imp2mat.toMat(img.getProcessor())
    ImMat_out = Mat()
    ImMat_real = Mat()
    dft(ImMat, ImMat_out, DFT_COMPLEX_OUTPUT, 0)
    dft(ImMat_out, ImMat_real, DFT_INVERSE + DFT_REAL_OUTPUT, 0)
    
    NewIP  = mat2ip.toImageProcessor(ImMat_real)
    ret = NewIP.getPixels()
    
    return ret
    
    
    
    
    

def split_list(alist, wanted_parts=1):
    """Split a list to the given number of parts."""
    length = len(alist)
    # alist[a:b:step] is used to get only a subsection of the list 'alist'.
    # alist[a:b] is the same as [a:b:1].
    # '//' is an integer division.
    # Without 'from __future__ import division' '/' would be an integer division.
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
             for i in range(wanted_parts) ]

#start_time = time.time()

# Part2 - Set up initial parameters
default_dz = 1                                      #This is the distance between the out of focus image and the focal plane
default_Lambda = 485                               #Wavelength in microns
#pi = 3.1415926535
pi = math.pi
#k0 = (2 * pi)/(default_Lambda)                      #Wavenumber
default_hp = 0.1073                                        #Pixel width in microns
#default_globaloffset =0.1                            #scaling factor (amplitude) for Hanning bump (alpha in equation (5))
#default_w0 = 2.2                                     #width of the Hanning bump (w0 in equation (5) and (6))
#default_n= 1.34

#Input sample images below. Three images are required, a focal plane image, I0, and two offset images Ia and Ib
#Ia is the ‘above’ image, Ib is the 'below' image, I0 is the in focus image, myBG is the background image
#Ia = 1.0*(imread('above.tif'))
#Ib = 1.0*(imread('below.tif'))
#I0 = 1.0*(imread('focus.tif'))
#BG = 1.0*(imread('bg.tif'))
guiPlus = GenericDialogPlus("An enhanced GenericDialog")
#gui = GenericDialog("TIE analysis")
guiPlus.addNumericField("dz (distance): ", default_dz, 3)
guiPlus.addNumericField("Lambda (wavelength) x1e-9: ", default_Lambda, 4)
guiPlus.addNumericField("Pixel width: ", default_hp, 4)
#guiPlus.addNumericField("Global offset (amplitude):", default_globaloffset, 4)
#guiPlus.addNumericField("w0 (width of the Hanning bump):", default_w0, 4)
#guiPlus.addNumericField("n: ", default_n, 4)
guiPlus.addImageChoice("Overfocused Image", "Image1")
guiPlus.addImageChoice("Underfocused Image", "Image2")
#guiPlus.addImageChoice("Focused Image", "Image3")
#guiPlus.addImageChoice("Background", "Image4")

guiPlus.addCheckbox("Continue Next Round",  False)
#guiPlus.addCheckbox("Use Default Background",  False)
#guiPlus.addDirectoryField("CSV Save Directory", "./")
#gui.showDialog()

while True:
    guiPlus.showDialog()
    if guiPlus.wasCanceled():
        break
    if guiPlus.wasOKed():
        dz = guiPlus.getNextNumber()
        Lambda = guiPlus.getNextNumber() * 1e-9
        #globaloffset = guiPlus.getNextNumber()
        #k0 = (2 * pi)/(Lambda)
        hp = guiPlus.getNextNumber()
        h = hp * 1e-6
        delz = dz * 1e-6 * 2
        #w0 = guiPlus.getNextNumber()
        #n = guiPlus.getNextNumber()

        Ia = guiPlus.getNextImage()
        Ib = guiPlus.getNextImage()
        #I0 = guiPlus.getNextImage()
       
        doOption = guiPlus.getNextBoolean()
        #directory = guiPlus.getNextString()


        Ia_pixels = Ia.getProcessor().getPixels()
        Ib_pixels = Ib.getProcessor().getPixels()
        #I0_pixels = Ia.getProcessor().getPixels()
        #I0_pixels = I0.getProcessor().getPixels()
        #BG_pixels = BG.getProcessor().getPixels()
        #dI = BG.getProcessor().getPixels()
        #print(len(dI))
        
        Dimx = Ia.getProcessor().getWidth()
        Dimy = Ia.getProcessor().getHeight()
        #Dimx = len(dI[0,:])          #dimension of the input images in the x direction
        #Dimy = len(dI[:,0])          #dimension of the input images in the y direction
        print("Dimx:", Dimx)
        print("Dimy:", Dimy)
        
        # Make the two images to be square
        
        #if Dimx != Dimy:
            #print("The two images need to be square!")
            #sys.exit(1)
            #Dim = Dimx
        
        Dimx1 = Ib.getProcessor().getWidth()
        Dimy1 = Ib.getProcessor().getHeight()
        if Dimx1 != Dimx or Dimy1 != Dimy:
            print("The two images need to be of the same size!")
            sys.exit(2)
        
        progress = 0
        IJ.showProgress(0)
        
        #Dim = Dimx
            
        Ia1 = [0] * Dimx * Dimy
        Ib1 = [0] * Dimx * Dimy
        for i in range(Dimy):
            for j in range(Dimx):
                Ia1[i * Dimx + j] = float(Ia_pixels[i * Dimx + j])
        for i in range(Dimy):
            for j in range(Dimx):
                Ib1[i * Dimx + j] = float(Ib_pixels[i * Dimx + j])
        #Dimx = Dim
        #Dimy = Dim
        
        # Adjust Brightness
        sz = Dimx * Dimy
        avg0 = 0
        avg1 = 0
        for i in range(sz):
            avg0 = avg0 + Ia1[i]/sz
        for i in range(sz):
            avg1 = avg1 + Ib1[i]/sz
        avg2 = (avg0 + avg1)/2
        for i in range(sz):
            Ia1[i] = Ia1[i] + (avg2 - avg0)
        for i in range(sz):
            Ib1[i] = Ib1[i] + (avg2 - avg1)

    #Part3 - Transport of intensity equation adapted from Davis
       
    #Calculate first derivative of intensity with respect to defocus
    #This is equivalent to dI/dz in the paper in equation 1
    #Empiracally calculated from measured values so the derivative does not need to be calculated symbolically

        #assert len(Ia_pixels) > 0, "Fail to read the Overfocused Image!"
        #assert len(Ib_pixels) > 0, "Fail to read the Underfocused Image!"
        #assert len(I0_pixels) > 0, "Fail to read the Focused Image!"
        #assert len(BG_pixels) > 0, "Fail to read the Background!"

        #dI = array.array('f')
        #dI.extend([0.1])
        dI = [0] * len(Ia1)
        print(len(dI))
        print(len(Ia1))
        k0 = (-2 * pi) / Lambda
        for i in range(len(Ia1)):
            dI[i] = (Ia1[i] - Ib1[i]) / delz * k0
            #dI.append(0.1)
       
        #time.sleep(2)
       
        print(len(dI))
        print(dI[0], Ia1[0] - Ib1[0], delz)


        #width = Ia.getProcessor().getWidth()
        #print(width)
        #dI = (k0  * (Ia_pixels-Ib_pixels)/(dz))

    #This section is coordinates and matrix sizes. It converts from the spatial dimensions of the input images into
    #frequency units for the Fourier Domain
    #Image dimensions

        print(dI[0:10])
        phase_est1 = fft_poisson(dI, h, Dimx, Dimy)
        print(phase_est1[0:10])
        
        dc1 = [0] * (Dimx - 1) * Dimy
        dr1 = [0] * Dimx * (Dimy - 1)
        for i in range(Dimy):
            for j in range(Dimx - 1):
                dc1[i * (Dimx - 1) + j] = (phase_est1[i * Dimx + j + 1] - phase_est1[i * Dimx + j]) / Ia1[i * Dimx + j + 1]        
        for i in range(Dimy - 1):
            for j in range(Dimx):
                dr1[i * Dimx + j] = (phase_est1[(i + 1) * Dimx + j] - phase_est1[i * Dimx + j]) / Ia1[(i + 1) * Dimx + j]

        dc2 = [0] * (Dimx - 2) * Dimy
        dr2 = [0] * Dimx * (Dimy - 2)
        for i in range(Dimy):
            for j in range(Dimx - 2):
                dc2[i * (Dimx - 2) + j] = dc1[i * (Dimx - 1) + j + 1] - dc1[i * (Dimx - 1) + j]
        for i in range(Dimy - 2):
            for j in range(Dimx):
                dr2[i * Dimx + j] = dr1[(i + 1) * Dimx + j] - dr1[i * Dimx + j]
                
        aux = [0] * (Dimx - 2) * (Dimy - 2)
        for i in range(Dimy - 2):
            for j in range(Dimx - 2):
                aux[i * (Dimx - 2) + j] = dr2[i * Dimx + j] + dc2[i * (Dimx - 2) + j]
        h = 1
        print(aux[0:10])
        ph_est = fft_poisson(aux, h, Dimx - 2, Dimy - 2)
        print(ph_est[0:10])
        
        print("Finished!")

        IJ.showProgress(1)

    #Part 5 - Display TIE and RI images, save images, and save csv files.

        Dimx = Dimx - 2
        Dimy = Dimy - 2

        #ph_high = high2d(ph_est, Dimy, 0.005)
        #ph_high = ph_est

        name0 = Ia.getTitle()
        name1 = name0.rsplit(".", 1)

        pixel_matrix01 = split_list(ph_est, wanted_parts = Dimy)
        pixel_matrix01 = [list(x) for x in zip(*pixel_matrix01)]
        TIE = ImagePlus(name1[0] + "_tie." + name1[1], FloatProcessor(pixel_matrix01))
        TIE.show()
        #IJ.run("Rotate 90 Degrees Left")
        #IJ.run("Flip Vertically")
        
        if doOption == False:
            break
        

        '''
        pixel_matrix02 = split_list(ph_high, wanted_parts = Dimy)
        RefractiveIndex = ImagePlus("Filtered TIE phase map", FloatProcessor(pixel_matrix02))
        RefractiveIndex.show()
        IJ.run("Rotate 90 Degrees Left")
        IJ.run("Flip Vertically")
        '''
        
        '''
        # Test

        file1 = open(directory + "/TIE_.csv", "w")
        for i in range(0, Dimy):
            for j in range(0, Dimx - 1):
                file1.write(str(ph_est[i * Dimx + j] * 1000 + 32000) + ",")
            file1.write(str(ph_est[i * Dimx + Dimx - 1] * 1000 + 32000) + "\n")
        file1.close()

        file2 = open(directory + "/RI_.csv", "w")
        for i in range(0, Dimy):
            for j in range(0, Dimx - 1):
                file2.write(str(ph_high[i * Dimx + j] * 1000 + 32000) + ",")
            file2.write(str(ph_high[i * Dimx + Dimx - 1] * 1000 + 32000) + "\n")
        file2.close()
        '''


#Updates:
#3/12/2024  Enable to process non-squared images. Fixed the progress bar.
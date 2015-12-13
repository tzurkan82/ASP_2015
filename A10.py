#
#
#   sineModel 
#   with multiresolution alalysis and the original version without it
#
#   This code has been testen on Windows 8 within Spyder (Python 2.7) environment only
#

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))

import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.signal import get_window
from scipy.fftpack import ifft, fftshift
import math
import dftModel as DFT
import utilFunctions as UF
import matplotlib.pyplot as plt

import sineModel
  
  
 
 
 
# 
# 
#
#   Selecting the peaks from the 3 frequency bands defined by B1,B2, and B3. 
#   The bands defined the following way:
#       1st band - from 0 Hz to B1 - 1 Hz,
#       2nd band - from B1 Hz to B2 - 1 Hz,
#       3rd band - from B2 Hz to B3 - 1 Hz
#
#   Also the function deletes duplicate frequencies on the boundaries between the bands:
#   if the two given frequencies on the boundary is within 2 hertz distance, one (the lower) is deleted.
#
#
#
def selectPeaks(ipfreq1, ipmag1, ipphase1, ipfreq2, ipmag2, ipphase2, ipfreq3, ipmag3, ipphase3, B1, B2, B3):
    
    ipfreq = np.zeros(0)    # Output combined arrays of frequencies, magnitudes, and phases from the 3 bands
    ipmag = np.zeros(0)
    ipphase = np.zeros(0)
    
    boundary_flag = False
       
    i = 0
    n = ipfreq1.size
    while i < n:
        f = ipfreq1[i]
        if (f < B1):
            ipfreq = np.append(ipfreq, f)
            ipmag = np.append(ipmag, ipmag1[i])
            ipphase = np.append(ipphase, ipphase1[i])
            i += 1
        else:
            i = n # exit the loop
            
    i = 0
    j = 0
    n = ipfreq2.size
    while i < n:
        f = ipfreq2[i]
        if (f < B1):
            i += 1
            boundary_flag = True  
        elif (f < B2):
            #print B2
            #print f   
            if (boundary_flag):
                if (i > 0):
                    # Checking for intersection of frequencies from different bands
                    #print "Checking for intersection (around B1)"
                    #print f
                    #print ipfreq[ipfreq.size-1]
                    if (np.floor(f) == np.ceil(ipfreq[ipfreq.size-1])):
                        #print "Found intersection of frequencies. Deleting one instance"
                        #print ipfreq
                        ipfreq = np.delete(ipfreq, ipfreq.size-1, None)  
                        ipmag = np.delete(ipmag, ipmag.size-1, None) 
                        ipphase = np.delete(ipphase, ipphase.size-1, None) 
                        #print ipfreq
                        boundary_flag = False
                    else:
                        #print "No intersection"
                        boundary_flag = False
                        
            ipfreq = np.append(ipfreq, f)
            ipmag = np.append(ipmag, ipmag2[i])
            ipphase = np.append(ipphase, ipphase2[i])
            i += 1
        else:
            i = n #exit the loop
            
    i = 0
    j = 0
    n = ipfreq3.size
    while i < n:
        f = ipfreq3[i]
        if (f < B2):
            i += 1
            boundary_flag = True
        elif (f < B3):
            if (boundary_flag):
                if (i > 0):
                    # Checking for intersection of frequencies from different bands
                    #print "Checking for intersection (around B2)"
                    #print f
                    #print ipfreq[ipfreq.size-1]
                    if (np.floor(f) == np.ceil(ipfreq[ipfreq.size-1])):
                        print "Found intersection of frequencies. Deleting one instance"
                        print ipfreq
                        ipfreq = np.delete(ipfreq, ipfreq.size-1, None)
                        ipmag = np.delete(ipmag, ipmag.size-1, None) 
                        ipphase = np.delete(ipphase, ipphase.size-1, None) 
                        print ipfreq
                        boundary_flag = False
                    else:
                        #print "No intersection"
                        boundary_flag = False
            
            ipfreq = np.append(ipfreq, f)
            ipmag = np.append(ipmag, ipmag3[i])
            ipphase = np.append(ipphase, ipphase3[i])
            i += 1
        else:
            i = n #exit the loop
            
            
            
    #ipphase = ipphase1
    
        
    return ipfreq, ipmag, ipphase







#
#
#
#   The analysis/synthesis of a sound using the sinusoidal model (without sine tracking) 
#   using multiresolution analysis: there are 3 bands, defined by B1, B2, abd B3, 
#   and each band is analyzed with its own parameters values: with windows and FFTs of different size.
#   All this is done in order to capture more different low frequencies, and at the same time, capture short-time details (changes in time) 
#   in the high frequency region.
#
#   fs should be 44100 Hz normally.
#   B1 must be lower than B2, and B2 must be lower than B3.
#   B3 should be 22050 Hz normally.
#   N1 and w1.size must be higher than N2 and w2.size, 
#   and N2 and w2.size must be higher than N3 and w3.size.
#   N1, N2, and N3 must be a power of 2, greater than the corresponding sizes of w1, w2, and w3.
#
#   See sineModelMultiResTest1() function below for the sample proper parameters values.
#
#
#
def sineModelMultiRes(x, fs, w1, w2, w3, N1, N2, N3, t, B1, B2, B3):
    
    if (B3 > (44100 / 2)):
        B3 = 22050

    Ns = 512                                                # FFT size for synthesis (even)
    H = Ns/4                                                # Hop size used for analysis and synthesis
    hNs = Ns/2                                              # half of synthesis FFT size

    # ------ The first analysis values for the frequency band #1 (B1) --------
    hM11 = int(math.floor((w1.size+1)/2))                   # half analysis window size by rounding
    hM12 = int(math.floor(w1.size/2))                       # half analysis window size by floor
    pin1 = max(hNs, hM11)                                    # init sound pointer in middle of anal window 
    print "\nThe analysis parameters for the band #1:"
    print "N1"
    print N1
    print "Half analysis window size by rounding (hM11):"
    print hM11
    print "pin 1:"
    print pin1      
    pend1 = x.size - max(hNs, hM11)                          # last sample to start a frame
    print "pend 1:"
    print pend1
    fftbuffer = np.zeros(N1)                                # initialize buffer for FFT
    w1 = w1 / sum(w1)                                       # normalize analysis window
    
    # ------ The second analysis for the frequency band #2 (B2) --------
    hM21 = int(math.floor((w2.size+1)/2))                   # half analysis window size by rounding
    hM22 = int(math.floor(w2.size/2))                       # half analysis window size by floor
    pin2 = max(hNs, hM21)  
    print "\nThe analysis parameters for the band #2:"
    print "N2"
    print N2
    print "Half analysis window size by rounding (hM21):"
    print hM21                                     
    print "pin 2:"
    print pin2 
    pend2 = x.size - max(hNs, hM21)                          # last sample to start a frame
    print "pend 2:"
    print pend2
    fftbuffer = np.zeros(N2)                                # initialize buffer for FFT
    w2 = w2 / sum(w2)                                       # normalize analysis window
    
    # ------ The third analysis for the frequency band #3 (B3) --------
    hM31 = int(math.floor((w3.size+1)/2))                   # half analysis window size by rounding
    hM32 = int(math.floor(w3.size/2))                       # half analysis window size by floor
    pin3 = max(hNs, hM31)                                    # init sound pointer in middle of anal window 
    print "\nThe analysis parameters for the band #3:"
    print "N3"
    print N3
    print "Half analysis window size by rounding (hM31):"
    print hM31
    print "pin 3:"
    print pin3       
    pend3 = x.size - max(hNs, hM31)                          # last sample to start a frame
    print "pend 3:"
    print pend3
    print "\n"
    fftbuffer = np.zeros(N3)                                # initialize buffer for FFT
    w3 = w3 / sum(w3)                                       # normalize analysis window
    
    #----- The synthesis parameters -----    
    hM1 = hM11                                              # half analysis window size by rounding
    hM2 = hM12                                              # half analysis window size by floor
    print "hM1, hM2:"
    print hM1, hM2
    pin = max(hNs, hM1)                                     # init sound pointer in middle of anal window       
    pend = x.size - max(hNs, hM1)                           # last sample to start a frame
    print "pin, pend:"
    print pin
    print pend
    yw = np.zeros(Ns)                                       # initialize output sound frame
    y = np.zeros(x.size)                                    # initialize output array
    sw = np.zeros(Ns)                                       # initialize synthesis window
    ow = triang(2*H)                                        # triangular window
    sw[hNs-H:hNs+H] = ow                                    # add triangular window
    bh = blackmanharris(Ns)                                 # blackmanharris window
    bh = bh / sum(bh)                                       # normalized blackmanharris window
    sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window
    
    #------- Analyze the sound "x" in the 3 subfrequency bands and synthesize the sound "y" -----------
    
    frame = 0
    
    while pin1<pend1:                                         # while input sound pointer is within sound
    
        #print "The current frame is: "
        #print frame
    
        xt1 = x.copy()
        xt2 = x.copy()
        xt3 = x.copy() 
        
        pin2 = pin1
        pin3 = pin1
        
        x1 = xt1[pin1-hM11:pin1+hM12]                         # select frame
        mX1, pX1 = DFT.dftAnal(x1, w1, N1)                      # compute dft
        ploc1 = UF.peakDetection(mX1, t)                        # detect locations of peaks
        #pmag = mX[ploc]                                       # get the magnitude of the peaks
        iploc1, ipmag1, ipphase1 = UF.peakInterp(mX1, pX1, ploc1)# refine peak values by interpolation
        ipfreq1 = fs*iploc1/float(N1)                         # convert peak locations to Hertz

   
        x2 = xt2[pin2-hM21:pin2+hM22]                         # select frame

        #x2 = x1        
        
        mX2, pX2 = DFT.dftAnal(x2, w2, N2)                      # compute dft
        ploc2 = UF.peakDetection(mX2, t)                        # detect locations of peaks
        #pmag = mX[ploc]                                       # get the magnitude of the peaks
        iploc2, ipmag2, ipphase2 = UF.peakInterp(mX2, pX2, ploc2)# refine peak values by interpolation
        ipfreq2 = fs*iploc2/float(N2)                         # convert peak locations to Hertz
        
    
        x3 = xt3[pin3-hM31:pin3+hM32]                         # select frame
        mX3, pX3 = DFT.dftAnal(x3, w3, N3)                      # compute dft
        ploc3 = UF.peakDetection(mX3, t)                        # detect locations of peaks
        #pmag = mX[ploc]                                       # get the magnitude of the peaks
        iploc3, ipmag3, ipphase3 = UF.peakInterp(mX3, pX3, ploc3)# refine peak values by interpolation
        ipfreq3 = fs*iploc3/float(N3)                         # convert peak locations to Hertz
        
        ipfreq, ipmag, ipphase = selectPeaks(ipfreq1, ipmag1, ipphase1, ipfreq2, ipmag2, ipphase2, ipfreq3, ipmag3, ipphase3, B1, B2, B3)
        
        #print "Synthesis window size:"
        #print M

        Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)   # generate sines in the spectrum         
        fftbuffer = np.real(ifft(Y))                          # compute inverse FFT
        yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
        yw[hNs-1:] = fftbuffer[:hNs+1] 
        y[pin1-hNs:pin1+hNs] += sw*yw                         # overlap-add and apply a synthesis window
        
        pin1 += H                                              # advance sound pointer
        frame += 1
        
        
    print "\n\n----- STATISTICS FOR THE LAST FRAME (for debug purposes, etc) -----\n"
    print "The last frame frequencies values:"
    print ipfreq1
    print ipfreq2
    print ipfreq3
    print "The last frame magnitudes values:"
    print ipmag1
    print ipmag2
    print ipmag3
    print "The last frame phases values:"
    print ipphase1
    print ipphase2
    print ipphase3
    """
    print "mX1:"
    print mX1[0:64]
    print "mX2:"
    print mX2[0:64]
    print "mX3:"
    print mX3[0:64]
    """
    print "\n"
    print "Synthesizing the following frequencies for the last frame:"
    print ipfreq
    print "Synthesizing the following magnitudes for the last frame:"
    print ipmag
    print "Synthesizing the following phases for the last frame:"
    print ipphase
    print "\n"
    print "Total frames analyzed/synthesized:"
    print frame
    print "\n\n"

    UF.wavwrite(y, fs, "./OutFile(sineModel_MultiresAnalysis).wav")
    #UF.wavwrite(y, fs, "c:/OutFile(sineModel_MultiresAnalysis).wav")
  
    return y
 
 
 
 
 
 
#
#                                   
#
#   Test function: sineModelOriginalTest1
#      (calls the ORIGINAL sineModel FUNCTION (from sineModel.py))
#
#
#
def sineModelOriginalTest1(inputFile, M):
    
    print "\n\n\n###############  RUN THE ORIGINAL TEST (without multiresolution)  ###############\n"
    
    #M1 = 4095
    M1 = M
    
    print "M: "
    print M
    
    N1 = int(pow(2, np.ceil(np.log2(M1))))      # FFT Size, power of 2 larger than M

    print "N1: "
    print N1
    
    t = -60.0                                   # threshold

    fs, x = UF.wavread(inputFile)               # read input sound
    
    #print "Ploting \"x\""
    #plt.plot(x)
    
    window = 'blackman'                         # Window type
    w1 = get_window(window, M1)                 # compute analysis window
            
    return sineModelOriginal(x,fs,w1,N1,t)
    
    
    
    
    
    
    
    
#
#
#
#   Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
#       (the ORIGINAL from sineModel.py, slightly modified to show some debug information)
#
#
#
def sineModelOriginal(x, fs, w, N, t):
    """
    Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
    x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB 
    returns y: output array sound
    """

    hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
    hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
    Ns = 512                                                # FFT size for synthesis (even)
    H = Ns/4                                                # Hop size used for analysis and synthesis
    hNs = Ns/2                                              # half of synthesis FFT size
    pin = max(hNs, hM1)                                     # init sound pointer in middle of anal window       
    pend = x.size - max(hNs, hM1)                           # last sample to start a frame
    fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
    yw = np.zeros(Ns)                                       # initialize output sound frame
    y = np.zeros(x.size)                                    # initialize output array
    w = w / sum(w)                                          # normalize analysis window
    sw = np.zeros(Ns)                                       # initialize synthesis window
    ow = triang(2*H)                                        # triangular window
    sw[hNs-H:hNs+H] = ow                                    # add triangular window
    bh = blackmanharris(Ns)                                 # blackmanharris window
    bh = bh / sum(bh)                                       # normalized blackmanharris window
    sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window
     
    frame = 0
    while pin<pend:                                         # while input sound pointer is within sound 
       
	#-----analysis-----    
# pin - the center of the analysis window (analysis window is always a power of 2, no less than hNs(512/2))
# pend = x.size - pin         
       x1 = x[pin-hM1:pin+hM2]                               # select frame
       mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
       ploc = UF.peakDetection(mX, t)                        # detect locations of peaks
       pmag = mX[ploc]                                       # get the magnitude of the peaks
       iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
       ipfreq = fs*iploc/float(N)                            # convert peak locations to Hertz
       Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)   # generate sines in the spectrum         
       fftbuffer = np.real(ifft(Y))                          # compute inverse FFT
       yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
       yw[hNs-1:] = fftbuffer[:hNs+1] 
       y[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window
  
# H is always fixed here
       pin += H                                              # advance sound pointer
       frame += 1
        
        
    print "\n\n----- STATISTICS FOR THE LAST FRAME (for debug purposes, etc) -----\n"
    print "\n"
    print "Synthesizing the following frequencies for the last frame:"
    print ipfreq
    print "Synthesizing the following magnitudes for the last frame:"
    print ipmag
    print "Synthesizing the following phases for the last frame:"
    print ipphase
    print "\n"
    print "Total frames analyzed/synthesized:"
    print frame
    print "\n\n"
    
    UF.wavwrite(y, fs, "./OutFile(sineModel_OriginalAnalysis).wav")
    #UF.wavwrite(y, fs, "c:/OutFile(sineModel_OriginalAnalysis).wav")
    
    return y
    
    
    



#
#    
#
#   Test function: sineModelMultiResTest1
#     (calls MODIFIED FUNCTION sineModelMultiRes FOR MULTIRESOLUTION ANALYSIS)
#
#
#
def sineModelMultiResTest1(inputFile, M1, M2, M3):
    
    print "\n\n\n###############  RUN MULTIRESOLUTION TEST  ###############\n"
    
    #M1 = 4095
    #M2 = 2047
    #M3 = 1023
    
    print "M1: "
    print M1
    print "M2: "
    print M2
    print "M3: "
    print M3
    
    N1 = int(pow(2, np.ceil(np.log2(M1))))      # FFT Size, power of 2 larger than M
    N2 = int(pow(2, np.ceil(np.log2(M2))))      # FFT Size, power of 2 larger than M
    N3 = int(pow(2, np.ceil(np.log2(M3))))      # FFT Size, power of 2 larger than M
    
    print "N1: "
    print N1
    print "N2: "
    print N2
    print "N3: "
    print N3
    
    t = -60.0                                   # threshold

    fs, x = UF.wavread(inputFile)               # read input sound
    
    #print "Ploting \"x\""
    #plt.plot(x)
    
    window = 'blackman'                         # Window type
    w1 = get_window(window, M1)                 # compute analysis window
    w2 = get_window(window, M2)                 # compute analysis window
    w3 = get_window(window, M3)                 # compute analysis window
    
    #B1 = 1000    # the band from 0 to 999 Hz
    #B2 = 5000    # the band from 1000 to 4999 Hz
    #B3 = 22050   # the band from 5000 to 22049 Hz
    B1 = 100
    B2 = 1000
    B3 = 22050
    
    if (B3 > (44100 / 2)):
        B3 = 22050
    
    return sineModelMultiRes(x, fs, w1, w2, w3, N1, N2, N3, t, B1=1000, B2=5000, B3=22050)
    
    




#
#
#
#   Compare the two versions of sineModel: 
#   modified with multiresolution analysis and original with single-resolution analysis
#
#
#
def sineModelRunComparisonTest(inputFile, M=4095, M2=2047, M3=1023):
    y1 = sineModelMultiResTest1(inputFile, M, M2, M3)
    y2 = sineModelOriginalTest1(inputFile, M)
    return y1, y2
    
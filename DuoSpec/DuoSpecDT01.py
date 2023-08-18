"""
   DuoSpecDT v01
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import math

def truncate( number, decimals=0 ) :
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance( decimals, int ) :
        raise TypeError( "decimal places must be an integer." )
    elif decimals < 0 :
        raise ValueError( "decimal places has to be 0 or more." )
    elif decimals == 0 :
        return math.trunc( number )
    factor = 10.0 ** decimals
    return math.trunc( number * factor ) / factor


filename1 = "Data/Table_vTitanium-20230721-14.01.txt"

filename2 = "Data/Table_xSilver-20230721-14.01.txt"

# digital re-sampling time interval
dgresamp = 0.002

testdat1 = np.loadtxt( filename1 )
tm1 = testdat1[:,1]
xacc1 = testdat1[:,2]
yacc1 = testdat1[:,3]
zacc1 = testdat1[:,4]

#nrec1 = tm1.size
ts1 = [ tm1[0] ]
te1 = [ tm1[-1] ]


testdat2 = np.loadtxt( filename2 )
tm2 = testdat2[:,1]
xacc2 = testdat2[:,2]
yacc2 = testdat2[:,3]
zacc2 = testdat2[:,4]

#nrec2 = tm2.size
ts2 = [ tm2[0] ]
te2 = [ tm2[-1] ]


startcut = max( np.concatenate((ts1, ts2)) )
endcut = min( np.concatenate((te1, te2)) )

rst = truncate( startcut, 3 ) + dgresamp
ret = truncate( endcut, 3 )

itm = np.arange( rst, ret, dgresamp )

ixacc1 = np.interp( itm, tm1, xacc1 )
iyacc1 = np.interp( itm, tm1, yacc1 )
izacc1 = np.interp( itm, tm1, zacc1 )

ixacc2 = np.interp( itm, tm2, xacc2 )
iyacc2 = np.interp( itm, tm2, yacc2 )
izacc2 = np.interp( itm, tm2, zacc2 )


# threshold amplitude to show phase angles of significant amplitude peaks
thas1 = 0.00009
thas2 = 0.00009
#thas1 = 0.00200
#thas2 = 0.00200
#thas1 = 0.00050
#thas2 = 0.00050

N = itm.size

Fx = fftpack.fft( ixacc1 )
fx = fftpack.fftfreq( N, dgresamp )
maskx = np.where( fx >= 0 )
pax = np.angle( Fx )
pax[np.where((np.abs(Fx) /(N//2)) < thas1)] = 0.


Fy = fftpack.fft( iyacc1 )
fy = fftpack.fftfreq( N, dgresamp )
masky = np.where( fy >= 0 )
pay = np.angle( Fy )
pay[np.where((np.abs(Fy)/(N//2)) < thas1)] = 0.


Fz = fftpack.fft( izacc1 )
fz = fftpack.fftfreq( N, dgresamp )
maskz = np.where( fz >= 0 )
paz = np.angle( Fz )
paz[np.where((np.abs(Fz)/(N//2)) < thas1)] = 0.


Fx2 = fftpack.fft( ixacc2 )
fx2 = fftpack.fftfreq( N, dgresamp )
maskx2 = np.where( fx2 >= 0 )
pax2 = np.angle( Fx2 )
pax2[np.where((np.abs(Fx2) /(N//2)) < thas2)] = 0.


Fy2 = fftpack.fft( iyacc2 )
fy2 = fftpack.fftfreq( N, dgresamp )
masky2 = np.where( fy2 >= 0 )
pay2 = np.angle( Fy2 )
pay2[np.where((np.abs(Fy2)/(N//2)) < thas2) ] = 0.


Fz2 = fftpack.fft( izacc2 )
fz2 = fftpack.fftfreq( N, dgresamp )
maskz2 = np.where( fz2 >= 0 )
paz2 = np.angle( Fz2 )
paz2[np.where((np.abs(Fz2)/(N//2)) < thas2)] = 0.


#xlimx = 25.
xlimx = 50.
ylimx = 0.00020
#ylimx = 0.0080

yt = [ -3.14, -1.57, 0., 1.57, 3.14 ]
ytlabel = [ r"$-\pi$", r"$-\pi/2$", "0", r"$+\pi/2$", r"$+\pi$" ]

plt.figure(11)

plt.subplot(4,1,1)
plt.stem( fx[maskx], abs(Fx[maskx])/(N//2), "c", markerfmt=".y", basefmt="-m" )
plt.xlim(0,xlimx)
plt.ylim(0,ylimx)
plt.ylabel( "Amplitude  ${|F\,|\,/\,(N//2)}$" )
plt.text(0,0,filename1)
plt.title("X-acc Amplitude and Phase Spectra")
plt.grid(axis='x')

plt.subplot(4,1,2)
plt.stem( fx[maskx], (pax[maskx]), 'c' )
plt.ylabel( "Phase Angle  (radian)" )
plt.text(0,0,filename1)
plt.xlim(0,xlimx)
plt.ylim(-4,4)
plt.yticks( yt, ytlabel )
plt.grid(axis='x')


plt.subplot(4,1,3)
plt.stem( fx2[maskx2], abs(Fx2[maskx2])/(N//2), "c", markerfmt=".y", basefmt="-m" )
plt.xlim(0,xlimx)
plt.ylim(0,ylimx)
plt.ylabel( "Amplitude  ${|F\,|\,/\,(N//2)}$" )
plt.text(0,0,filename2)
plt.grid(axis='x')


plt.subplot(4,1,4)
plt.stem( fx2[maskx2], (pax2[maskx2]), 'c' )
plt.ylabel( "Phase Angle  (radian)" )
plt.xlim(0,xlimx)
plt.xlabel("Frequency  (Hz)")
plt.ylim(-4,4)
plt.yticks( yt, ytlabel )
plt.text(0,0,filename2)
plt.grid(axis='x')

plt.show()


plt.figure(12)

plt.stem( fx[maskx], (pax[maskx]), 'b', markerfmt=".b",  label=filename1 )
plt.stem( fx2[maskx2], (pax2[maskx2]), 'r', markerfmt=".r", label=filename2 )
plt.xlabel("Frequency  (Hz)")
plt.ylabel( "Phase Angle  (radian)" )
plt.xlim(10,20)
plt.ylim(-4,4)
plt.yticks( yt, ytlabel )
plt.title("Phase comparison  (X-acc)")
plt.grid(axis='x')
plt.legend()

plt.show()

plt.figure(21)

plt.subplot(4,1,1)
plt.stem( fy[masky], abs(Fy[masky])/(N//2), "c", markerfmt=".y", basefmt="-m" )
plt.xlim(0,xlimx)
plt.ylim(0,ylimx)
plt.ylabel( "Spectral Amplitude  ${|F\,|\,/\,(N//2)}$" )
plt.text(0,0,filename1)
plt.title("Y-acc Amplitude and Phase Spectra")
plt.grid(axis='x')

plt.subplot(4,1,2)
plt.stem( fy[masky], (pay[masky]), 'c' )
plt.ylabel( "Phase Angle  (radian)" )
plt.text(0,0,filename1)
plt.xlim(0,xlimx)
plt.ylim(-4,4)
plt.yticks( yt, ytlabel )
plt.grid(axis='x')


plt.subplot(4,1,3)
plt.stem( fy2[masky2], abs(Fy2[masky2])/(N//2), "c", markerfmt=".y", basefmt="-m" )
plt.xlim(0,xlimx)
plt.ylim(0,ylimx)
plt.ylabel( "Spectral Amplitude  ${|F\,|\,/\,(N//2)}$" )
plt.text(0,0,filename2)
plt.grid(axis='x')


plt.subplot(4,1,4)
plt.stem( fy2[masky2], (pay2[masky2]), 'c' )
plt.ylabel( "Phase Angle  (radian)" )
plt.xlim(0,xlimx)
plt.xlabel("Frequency  (Hz)")
plt.ylim(-4,4)
plt.yticks( yt, ytlabel )
plt.text(0,0,filename2)
plt.grid(axis='x')


plt.show()


plt.figure(22)

plt.stem( fy[masky], (pay[masky]), 'b', markerfmt=".b",  label=filename1 )
plt.stem( fy2[masky2], (pay2[masky2]), 'r', markerfmt=".r", label=filename2 )
plt.xlabel("Frequency  (Hz)")
plt.ylabel( "Phase Angle  (radian)" )
plt.xlim(10,20)
plt.ylim(-4,4)
plt.yticks( yt, ytlabel )
plt.title("Phase comparison  (Y-acc)")
plt.grid(axis='x')
plt.legend()

plt.show()





plt.figure(31)

plt.subplot(4,1,1)
plt.stem( fz[maskz], abs(Fz[maskz])/(N//2), "c", markerfmt=".y", basefmt="-m" )
plt.xlim(0,xlimx)
plt.ylim(0,ylimx)
plt.ylabel( "Spectral Amplitude  ${|F\,|\,/\,(N//2)}$" )
plt.text(0,0,filename1)
plt.title("Z-acc Amplitude and Phase Spectra")
plt.grid(axis='x')

plt.subplot(4,1,2)
plt.stem( fz[maskz], (paz[maskz]), 'c' )
plt.ylabel( "Phase Angle  (radian)" )
plt.text(0,0,filename1)
plt.xlim(0,xlimx)
plt.ylim(-4,4)
plt.yticks( yt, ytlabel )
plt.grid(axis='x')


plt.subplot(4,1,3)
plt.stem( fz2[maskz2], abs(Fz2[maskz2])/(N//2), "c", markerfmt=".y", basefmt="-m" )
plt.xlim(0,xlimx)
plt.ylim(0,ylimx)
plt.ylabel( "Spectral Amplitude  ${|F\,|\,/\,(N//2)}$" )
plt.text(0,0,filename2)
plt.grid(axis='x')


plt.subplot(4,1,4)
plt.stem( fz2[maskz2], (paz2[maskz2]), 'c' )
plt.ylabel( "Phase Angle  (radian)" )
plt.xlim(0,xlimx)
plt.xlabel("Frequency  (Hz)")
plt.ylim(-4,4)
plt.yticks( yt, ytlabel )
plt.text(0,0,filename2)
plt.grid(axis='x')


plt.show()

plt.figure(32)

plt.stem( fz[maskz], (paz[maskz]), 'b', markerfmt=".b",  label=filename1 )
plt.stem( fz2[maskz2], (paz2[maskz2]), 'r', markerfmt=".r", label=filename2 )
plt.xlabel("Frequency  (Hz)")
plt.ylabel( "Phase Angle  (radian)" )
plt.xlim(10,20)
plt.ylim(-4,4)
plt.yticks( yt, ytlabel )
plt.title("Phase comparison  (Z-acc)")
plt.grid(axis='x')
plt.legend()

plt.show()



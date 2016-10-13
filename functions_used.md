# Functions Used

## 1 Load Data

settagpath()
loadprh()
uigetfile()

## 2 Define dives

finddives()

## 3 Separate low/high acc signals

speclev()
peakfinder()
runmean()
smoothS()
Ahf_Anlf()
a2pr()

## 4

smoothpitch() ?

## 5 Estimate swim speed

inst_speed()

## 6 Estimate SW density around whales

SWdensityFromCTD()
EstimateDsw()

## 7 Extract Strokes and Glides

Ahf_Anlf()
buffer() # matlab?

### 7.2 user body rotation estimated 

magnet_rot_sa()
eventon()?

## 8 Make 5sec sub-glides

splitGL() ?

## 9 create summary table

m2h()

## 10 Cacl glide desc / glide asc

regress()
circ_mean()
cirv_var()

## 11 Cacl glide ratio


# Glossary

S: amount of power in each frequency (f)
f: frequency within power spectra
FR
FR1
end_des: end of descent, first point after diving below min_dive_def where pitch positive
start_asc: last point before diving above min_dive_def at which pitch is negative

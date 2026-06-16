TITLE Sodium current 

COMMENT Equations from 
   Golomb D, Amitai Y (1997) Propagating neuronal discharges in
   neocortical slices: computational and experimental study. J Neurophys
   78: 1199-1211.

>< Gating kinetics are at 36 degC. 
ENDCOMMENT

NEURON {
        SUFFIX Na
        USEION na READ ena WRITE ina 
        RANGE g, ina
}

UNITS {
	(S)  = (siemens)
        (mA) = (milliamp)
        (mV) = (millivolt)
}

PARAMETER {
        g	(S/cm2)
}

ASSIGNED {
        v	(mV)
	ena	(mV)
        ina	(mA/cm2)
	mtau	(ms)
	htau	(ms)
	minf
	hinf
}

STATE { h }

BREAKPOINT { 
        SOLVE states METHOD cnexp
	ina= g* minf^3* h* (v- ena) 
}

DERIVATIVE states {
	rates()
	h'= (hinf- h)/ htau
}

INITIAL {
	rates()
	h= hinf 
}

PROCEDURE rates() { UNITSOFF
	minf= 1/ (1+ exp(-(v+ 30)/ 9.5))
	hinf= 1/ (1+ exp((v+ 53)/ 7))
	htau= 0.37+ 2.78/ (1+ exp((v+ 40.5)/ 6))
} UNITSON


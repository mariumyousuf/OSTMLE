
: NMDA channel. JM modified to allow for gmax. 

NEURON {
	POINT_PROCESS ExpSyn3
	RANGE tau, e, i, gmax
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau = 0.1 (ms) <1e-9,1e9>
	e = 0	(mV)
	gmax=0
	mg	= 1    (mM)		: external magnesium concentration
}

ASSIGNED {
	v (mV)
	i (nA)
}

STATE {
	g (uS)
}

INITIAL {
	g=0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = gmax*g*mgblock(v)*(v - e)
}

DERIVATIVE state {
	g' = -g/tau
}

NET_RECEIVE(weight (uS)) {
	state_discontinuity(g, g + weight)
}

FUNCTION mgblock(v(mV)) {
	TABLE 
	DEPEND mg
	FROM -140 TO 80 WITH 1000

	: from Jahr & Stevens

	mgblock = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))
}

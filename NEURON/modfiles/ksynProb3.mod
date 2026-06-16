COMMENT
alpha function synapse implemented as continuously integrated
kinetic scheme a la Srinivasan and Chiel (Neural Computation) so that
one can give many stimuli which summate.

Onset times are placed in the vector onset[SIZE]
Conductance located in state variable G
The amplitude of each individual alpha function is given by stim,
stim * t * exp(-t/tau).
The last onset time should be a very large number so stim stops getting
added to state A
JM: Included probabilistic release, facilitation and 1-2 short-term depression after Varela et al, J neuroscience
 




ENDCOMMENT

DEFINE SIZE 5000

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}


NEURON {
	POINT_PROCESS KSynProbFull
	RANGE tau, stim, e, i,onset, treleased, fromlist,from, gmax,  p,p_release,compart,ntimes,distance,synlocation,D0,F0,Dmag,Fmag, Ridx, NReleased
	RANGE C,C0, C1, C2, Ds, O, B
	RANGE gNMDA, gmax_nmda
	GLOBAL mg, Rb, Ru, Rd, Rr, Ro, Rc,rb
	
	NONSPECIFIC_CURRENT i
}


UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(umho) = (micromho)
	(mM) = (milli/liter)
	(uM) = (micro/liter)
}

PARAMETER {
	tau=.25 (ms)
	stim=50 (umho)
	e=0	(mV)
	v	(mV)
	gmax=0.1
	
	ntimes=0
	
	
	F0=1
	Fmag=0.1
	
	D0=1
	Dmag=0.1
	
	tauF=94		: from Varela et al.
	tauD1=400
	tauD2=9200
	
	from=-1
	compart=0	: do not change. assigned dynamically
	distance=0	: do not change. assigned dynamically
	synlocation=0.5	: do not change. assigned dynamically
	
	: Destexhe, Mainen & Sejnowski, 1996
	Rb	= 5e-3    (/uM /ms)	: binding 		
	Ru	= 12.9e-3  (/ms)	: unbinding		
	Rd	= 8.4e-3   (/ms)	: desensitization
	Rr	= 6.8e-3   (/ms)	: resensitization 
	Ro	= 46.5e-3   (/ms)	: opening
	Rc	= 73.8e-3   (/ms)	: closing
	
	gmax_nmda=0.001 (umho)
	mg	= 1    (mM)		: external magnesium concentration
}



COMMENT
	: Clements et al. 1992
	Rb	= 5e-3    (/uM /ms)	: binding 		
	Ru	= 9.5e-3  (/ms)	: unbinding		
	Rd	= 16e-3   (/ms)	: desensitization
	Rr	= 13e-3   (/ms)	: resensitization 
	Ro	= 25e-3   (/ms)	: opening
	Rc	= 59e-3   (/ms)	: closing

	: Hessler Shirke & Malinow 1993
	Rb	= 5e-3    (/uM /ms)	: binding 		
	Ru	= 9.5e-3  (/ms)	: unbinding		
	Rd	= 16e-3   (/ms)	: desensitization
	Rr	= 13e-3   (/ms)	: resensitization 
	Ro	= 25e-3   (/ms)	: opening
	Rc	= 59e-3   (/ms)	: closing

	: Clements & Westbrook 1991
	Rb	=  5    (uM /s)	: binding 		
	Ru	=  5	(/s)	: unbinding -> gives Kd = Rb/Ru = 1 uM
	Rd	=  4.0  (/s)	: desensitization
	Rr	=  0.3  (/s)	: resensitization 
	Ro	= 10  (/s)	: opening
	Rc	= 322   (/s)	: closing

	: Edmonds & Colquhoun 1992
	Rb	=  5    (uM /s)	: binding 		
	Ru	=  4.7  (/s)	: unbinding		
	Rd	=  8.4  (/s)	: desensitization
	Rr	=  1.8  (/s)	: resensitization 
	Ro	= 46.5  (/s)	: opening
	Rc	= 91.6  (/s)	: closing

	: Lester & Jahr 1992
	Rb	= 5    (uM /s)	: binding 		
	Ru	= 6.7   (/s)	: unbinding		
	Rd	= 15.2  (/s)	: desensitization
	Rr	= 9.4   (/s)	: resensitization 
	Ro	= 83.8  (/s)	: opening
	Rc	= 83.8  (/s)	: closing

ENDCOMMENT
ASSIGNED {
	
	index
	i (nA)
	bath (umho)
	k (/ms)
	
	onset[SIZE] (ms)
	Ridx
	treleased[SIZE]
	fromlist[SIZE]
	NReleased
	p
	p_release
	F
	D
	
	gNMDA 		(umho)		: conductance
	C 		(mM)		: pointer to glutamate concentration
	rb		(/ms)    : binding
	
}

STATE {
	A (umho)
	G (umho)
	: NMDA Channel states (all fractions)
	
	C0		: unbound
	C1		: single bound
	C2		: double bound
	Ds		: desensitized
	O		: open

	B		: fraction free of Mg2+ block
}

INITIAL {
	k = 1/tau
	A = 0
	G = 0
	index=0
	Ridx=0
	NReleased=0
	p=0
	F=F0
	D=D0
	p_release=0
	
	initvars()
	index=0
	ntimes=0
	
	rates(v)
	C0 = 1
	C=0
}

PROCEDURE initvars(){
	LOCAL i
	i=0
	while(i<SIZE){
		onset[i]=1e20
		treleased[i]=0
		fromlist[i]=-1
		i=i+1
	}
}


? current
BREAKPOINT {
	rates(v)
	SOLVE conductance
	
	gNMDA = gmax_nmda * O * B

	i = gmax*G*(v - e)+  gNMDA * (v - e)
}

: at each onset time a fixed quantity of material is added to state A
: this material moves through G with the form of an alpha function

PROCEDURE conductance() { 
	LOCAL ii
	C=0
	while(index < SIZE && t>onset[index]) {
		
		F=F0
		ii=0
		while(ii<index){
			F = F + Fmag*exp(-(t - onset[ii])/tauF)
			ii=ii+1
		}
		
		D=D0
		ii=0
		while(ii<Ridx){
			D = D - Dmag*exp(-(t - treleased[ii])/tauD1)	: use fast depression (tauD1)
			ii=ii+1
		}
		
		if(D<0){
			D=0
		}
		
		p_release=1-exp(-F*D)
		p = scop_random(0)		: uniform distribution between 0 and 1
		
		if(p<p_release){		: probabilistic release
		
			from=fromlist[index]
			VERBATIM
//	  		fprintf(stderr,"\n Realeased from %.0f onto %.0f at t=%.2f with weight %f",from, compart, t, gmax);
	  		ENDVERBATIM
	  	
			treleased[Ridx]=onset[index]
			Ridx=Ridx+1
			NReleased=NReleased+1
			A = stim		: unitary release
			C=70
		}else{
			A=0
			C=0

		}
		
		index=index+1
	}
	SOLVE state METHOD sparse
	
	VERBATIM
	return 0;
	ENDVERBATIM
}


? kinetics
KINETIC state {
	~ A <-> G	(k, 0)
	~ G <-> bath	(k, 0)
						:NMDA 5 states
	 rb = Rb * (1e3) * C
	~ C0 <-> C1	(rb,Ru)
	~ C1 <-> C2	(rb,Ru)
	~ C2 <-> Ds	(Rd,Rr)
	~ C2 <-> O	(Ro,Rc)

	CONSERVE C0+C1+C2+Ds+O = 1
}

PROCEDURE rates(v(mV)) {
	TABLE B
	DEPEND mg
	FROM -120 TO 100 WITH 200

	: from Jahr & Stevens

	B = 1 / (1 + exp(0.062 (/mV) * -v) * (mg / 3.57 (mM)))
}

